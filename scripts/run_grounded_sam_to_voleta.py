import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEVA_ROOT = ROOT / "vendor" / "Tracking-Anything-with-DEVA-main"
GROUNDING_DINO_ROOT = ROOT / "vendor" / "GroundingDINO-main"
SAM_ROOT = ROOT / "vendor" / "segment-anything-main"

for repo_path in (DEVA_ROOT, GROUNDING_DINO_ROOT, SAM_ROOT):
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

from deva.ext.grounding_dino import get_grounding_dino_model, segment_with_text  # noqa: E402


def build_config(args: argparse.Namespace) -> dict:
    deva_saves = DEVA_ROOT / "saves"
    return {
        "GROUNDING_DINO_CONFIG_PATH": str(deva_saves / "GroundingDINO_SwinT_OGC.py"),
        "GROUNDING_DINO_CHECKPOINT_PATH": str(deva_saves / "groundingdino_swint_ogc.pth"),
        "DINO_THRESHOLD": args.dino_threshold,
        "DINO_NMS_THRESHOLD": args.dino_nms_threshold,
        "sam_variant": "mobile",
        "MOBILE_SAM_CHECKPOINT_PATH": str(deva_saves / "mobile_sam.pt"),
    }


def load_image_rgb(image_path: Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"))


def save_binary_mask(mask_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text-guided GroundingDINO + MobileSAM per frame and export VolETA masks."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--dino-threshold", type=float, default=0.35)
    parser.add_argument("--dino-nms-threshold", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(DEVA_ROOT)

    image_paths = sorted(
        [p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise FileNotFoundError(f"No input images found under {args.images_dir}")

    cfg = build_config(args)
    gd_model, sam_model = get_grounding_dino_model(cfg, "cuda")
    prompts = [p.strip() for p in args.prompt.split(".") if p.strip()]

    print(f"Processing {len(image_paths)} frame(s) from {args.images_dir}")
    print(f"Prompt(s): {prompts}")
    print(f"Output dir: {args.output_dir}")

    total_start = time.perf_counter()
    per_frame_stats = []

    for index, image_path in enumerate(image_paths, start=1):
        frame_start = time.perf_counter()
        image_rgb = load_image_rgb(image_path)
        mask, segments_info = segment_with_text(
            config=cfg,
            gd_model=gd_model,
            sam=sam_model,
            image=image_rgb,
            prompts=prompts,
            min_side=args.min_side,
        )

        mask_np = mask.detach().cpu().numpy()
        out_name = f"{image_path.stem}_segmented_mask.png"
        out_path = args.output_dir / out_name
        save_binary_mask(mask_np, out_path)

        frame_time = time.perf_counter() - frame_start
        per_frame_stats.append(frame_time)
        print(
            f"[{index}/{len(image_paths)}] {image_path.name} -> {out_name} | "
            f"objects={len(segments_info)} | time={frame_time:.2f}s"
        )

    total_time = time.perf_counter() - total_start
    print(
        f"Done. Generated {len(image_paths)} mask(s) in {total_time:.2f}s "
        f"(avg={sum(per_frame_stats) / len(per_frame_stats):.2f}s/frame)."
    )


if __name__ == "__main__":
    main()
