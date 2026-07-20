import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def save_binary_mask(mask_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(output_path)


def _prompt_tokens(prompt: Optional[str], food: Optional[str]) -> List[str]:
    text = " ".join([prompt or "", food or ""]).lower()
    return [t.strip() for t in text.replace(".", " ").replace(",", " ").split() if t.strip()]


def _build_color_seed(image_rgb: np.ndarray, food: Optional[str], prompt: Optional[str]) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    tokens = set(_prompt_tokens(prompt, food))

    red = (((h <= 12) | (h >= 165)) & (s >= 45) & (v >= 45))
    yellow = ((h >= 18) & (h <= 45) & (s >= 70) & (v >= 110))
    green = ((h >= 35) & (h <= 90) & (s >= 40) & (v >= 45))
    brown = ((h >= 3) & (h <= 32) & (s >= 25) & (v >= 45) & (v <= 235))
    saturated = (s >= 38) & (v >= 45)

    if {"banana"} & tokens:
        seed = yellow
    elif {"strawberry", "tomato"} & tokens:
        seed = red
    elif {"apple"} & tokens:
        seed = red | green | yellow
    elif {"bread", "cake", "meat", "rice", "noodle", "food", "dish", "meal", "soup"} & tokens:
        seed = saturated | brown
    else:
        seed = saturated | brown

    return seed.astype(np.uint8)


def _clean_mask(mask: np.ndarray, image_shape: Tuple[int, int], max_components: int = 1) -> np.ndarray:
    h, w = image_shape
    min_area = max(80, int(0.0004 * h * w))
    max_area = int(0.45 * h * w)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    work = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    keep = np.zeros_like(work, dtype=np.uint8)
    components = []
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if x == 0 or y == 0 or x + bw >= w - 1 or y + bh >= h - 1:
            continue
        components.append((area, label))

    if not components:
        return keep

    # The food object is usually the largest non-border colored component after removing markers.
    components.sort(reverse=True)
    for _area, label in components[:max_components]:
        keep[labels == label] = 1

    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=2)
    keep = cv2.dilate(keep, kernel, iterations=1)
    return keep.astype(bool)


def _refine_with_grabcut(image_rgb: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    if not np.any(seed_mask):
        return seed_mask

    h, w = seed_mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sure_fg = cv2.erode(seed_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    prob_fg = cv2.dilate(seed_mask.astype(np.uint8), kernel, iterations=2).astype(bool)

    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[prob_fg] = cv2.GC_PR_FGD
    gc_mask[sure_fg] = cv2.GC_FGD

    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        refined = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        if np.count_nonzero(refined) > 0:
            return _clean_mask(refined.astype(np.uint8), (h, w), max_components=1)
    except cv2.error:
        pass
    return seed_mask


def _grabcut_from_box(image_rgb: np.ndarray, box_xyxy: List[float]) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy]
    x0 = max(0, min(x0, w - 2))
    y0 = max(0, min(y0, h - 2))
    x1 = max(x0 + 1, min(x1, w - 1))
    y1 = max(y0 + 1, min(y1, h - 1))

    rect_w = x1 - x0
    rect_h = y1 - y0
    rect = (x0, y0, rect_w, rect_h)
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(image_bgr, gc_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
    mask = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    return _clean_mask(mask.astype(np.uint8), (h, w), max_components=1)


def segment_with_opencv(
    image_path: Path,
    output_mask_path: Path,
    food: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict:
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    food_tokens = set(_prompt_tokens(None, food))
    multi_component = bool({"mixed", "platter", "soup", "rice", "noodle"} & food_tokens) and not bool(
        {"banana", "strawberry", "tomato", "apple"} & food_tokens
    )
    max_components = 3 if multi_component else 1
    seed = _build_color_seed(image_rgb, food=food, prompt=prompt)
    cleaned = _clean_mask(seed, image_rgb.shape[:2], max_components=max_components)
    refined = _refine_with_grabcut(image_rgb, cleaned)
    save_binary_mask(refined, output_mask_path)

    area = int(np.count_nonzero(refined))
    h, w = refined.shape
    ys, xs = np.where(refined)
    bbox = None
    if area > 0:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    return {
        "backend": "opencv_color_grabcut",
        "status": "ok" if area > 0 else "empty_mask",
        "mask": str(output_mask_path),
        "food": food,
        "prompt": prompt,
        "mask_area_px": area,
        "mask_area_ratio": float(area / max(h * w, 1)),
        "bbox_xyxy": bbox,
        "limitations": [
            "Fallback uses color and foreground heuristics, not semantic understanding.",
            "Works best for visually distinct single food items on simple backgrounds.",
            "GroundingDINO + SAM/MobileSAM should be used when model dependencies and checkpoints are available.",
        ],
    }


def segment_with_hf_groundingdino(
    image_path: Path,
    output_mask_path: Path,
    args: argparse.Namespace,
) -> Dict:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_id = getattr(args, "hf_detector_model", "IDEA-Research/grounding-dino-tiny")
    box_threshold = float(getattr(args, "box_threshold", 0.25))
    text_threshold = float(getattr(args, "text_threshold", 0.25))
    prompt = getattr(args, "prompt", None) or "food . dish . meal . fruit . rice . noodle . meat . bread . soup"
    food = getattr(args, "food", None)
    query = f"a {food}." if food else prompt

    image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results.get("boxes")
    scores = results.get("scores")
    labels = results.get("labels")
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("HuggingFace GroundingDINO found no box for the prompt.")

    best_idx = int(torch.argmax(scores).detach().cpu().item()) if scores is not None else 0
    best_box = boxes[best_idx].detach().cpu().tolist()
    best_score = float(scores[best_idx].detach().cpu().item()) if scores is not None else None
    best_label = labels[best_idx] if labels is not None else None
    mask = _grabcut_from_box(image_rgb, best_box)
    save_binary_mask(mask, output_mask_path)

    return {
        "backend": "hf_groundingdino_grabcut",
        "status": "ok",
        "model": model_id,
        "query": query,
        "mask": str(output_mask_path),
        "box_xyxy": best_box,
        "score": best_score,
        "label": best_label,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "note": "GroundingDINO provides the target box; GrabCut converts the box into a pixel mask.",
    }


def segment_with_grounded_sam(
    image_path: Path,
    output_mask_path: Path,
    args: argparse.Namespace,
) -> Dict:
    import torch
    from run_grounded_sam_to_voleta import (
        build_config,
        get_grounding_dino_model,
        load_image_rgb,
        save_binary_mask as save_model_mask,
        segment_with_text,
    )

    cfg = build_config(args)
    prompts = [p.strip() for p in args.prompt.split(".") if p.strip()]
    image_rgb = load_image_rgb(image_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    old_cwd = Path.cwd()
    try:
        os.chdir(ROOT / "vendor" / "Tracking-Anything-with-DEVA-main")
        gd_model, sam_model = get_grounding_dino_model(cfg, device)
        mask, segments_info = segment_with_text(
            config=cfg,
            gd_model=gd_model,
            sam=sam_model,
            image=image_rgb,
            prompts=prompts,
            min_side=args.min_side,
        )
    finally:
        os.chdir(old_cwd)

    mask_np = mask.detach().cpu().numpy()
    save_model_mask(mask_np, output_mask_path)
    return {
        "backend": "groundingdino_mobile_sam",
        "status": "ok",
        "mask": str(output_mask_path),
        "prompt": args.prompt,
        "segments_info": segments_info,
    }


def generate_auto_mask(
    image_path: Path,
    output_mask_path: Path,
    args: argparse.Namespace,
) -> Dict:
    backend = getattr(args, "auto_mask_backend", "auto")
    fallback_errors = []
    if backend in {"auto", "grounded_sam"}:
        try:
            return segment_with_grounded_sam(image_path, output_mask_path, args)
        except Exception as exc:
            if backend == "grounded_sam":
                raise
            fallback_errors.append(f"grounded_sam: {repr(exc)}")

    if backend in {"auto", "hf_groundingdino"}:
        try:
            return segment_with_hf_groundingdino(image_path, output_mask_path, args)
        except Exception as exc:
            if backend == "hf_groundingdino":
                raise
            fallback_errors.append(f"hf_groundingdino: {repr(exc)}")

    result = segment_with_opencv(
        image_path=image_path,
        output_mask_path=output_mask_path,
        food=getattr(args, "food", None),
        prompt=getattr(args, "prompt", None),
    )
    result["fallback_from"] = "grounded_sam/hf_groundingdino"
    result["fallback_reason"] = " | ".join(fallback_errors) if fallback_errors else None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a food mask without user-provided mask input.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--food", type=str)
    parser.add_argument("--prompt", type=str, default="food . dish . meal . fruit . rice . noodle . meat . bread . soup")
    parser.add_argument("--auto-mask-backend", choices=["auto", "grounded_sam", "hf_groundingdino", "opencv"], default="auto")
    parser.add_argument("--hf-detector-model", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--dino-threshold", type=float, default=0.35)
    parser.add_argument("--dino-nms-threshold", type=float, default=0.8)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_auto_mask(args.image, args.output_mask, args)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
