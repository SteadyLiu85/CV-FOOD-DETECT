import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import depth_pro
from build_noref_scaffold import build_from_depth


def append_log(log_path: Path, lines) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def save_depth_png_meters(depth_m: np.ndarray, out_path: Path, scale: float = 1000.0) -> None:
    depth_mm = np.clip(depth_m * scale, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm, mode="I;16").save(out_path)


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    work = depth.copy()
    valid = work > 0
    if mask is not None:
        valid = valid & mask
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    dmin = float(work[valid].min())
    dmax = float(work[valid].max())
    norm = np.zeros_like(work, dtype=np.float32)
    if dmax > dmin:
        norm[valid] = (work[valid] - dmin) / (dmax - dmin)
    color = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def draw_mask_bbox(image: Image.Image, mask: np.ndarray) -> Image.Image:
    canvas = image.copy()
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return canvas
    draw = ImageDraw.Draw(canvas)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=4)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a packaged DepthPro metric-depth demo for one food image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to DepthPro checkpoint .pt file")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--log-file", type=Path, default=Path("reports/noref_integration_log_20260511.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = args.output_dir / "01_process"
    res_dir = args.output_dir / "02_results"
    proc_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    mask_img = Image.open(args.mask).convert("L")
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.NEAREST)
    mask = np.array(mask_img) > args.mask_threshold

    shutil.copy2(args.image, proc_dir / "input_image.jpg")
    mask_img.save(proc_dir / "input_mask.png")
    draw_mask_bbox(image, mask).save(proc_dir / "input_with_mask_bbox.jpg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = depth_pro.depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT
    config.checkpoint_uri = str(args.checkpoint)
    model, transform = depth_pro.create_model_and_transforms(config=config, device=device)
    model.eval()

    rgb, _, f_px = depth_pro.load_rgb(args.image)
    rgb_t = transform(rgb)
    prediction = model.infer(rgb_t, f_px=f_px)
    pred = prediction["depth"].detach().cpu().numpy().astype(np.float32)
    focallength_px = float(prediction["focallength_px"])

    raw_min = float(pred.min())
    raw_max = float(pred.max())
    raw_mean = float(pred.mean())
    masked = np.where(mask, pred, 0.0)
    masked_valid = masked > 0
    masked_min = float(masked[masked_valid].min()) if np.any(masked_valid) else 0.0
    masked_max = float(masked[masked_valid].max()) if np.any(masked_valid) else 0.0
    masked_mean = float(masked[masked_valid].mean()) if np.any(masked_valid) else 0.0

    norm_all = pred - pred.min()
    if norm_all.max() > 0:
        norm_all = norm_all / norm_all.max()
    gray_all = (255 * norm_all).clip(0, 255).astype(np.uint8)
    Image.fromarray(gray_all).save(proc_dir / "depth_gray_full.png")

    if np.any(masked_valid):
        norm_mask = np.zeros_like(masked, dtype=np.float32)
        denom = masked_max - masked_min
        if denom > 0:
            norm_mask[masked_valid] = (masked[masked_valid] - masked_min) / denom
        gray_mask = (255 * norm_mask).clip(0, 255).astype(np.uint8)
    else:
        gray_mask = np.zeros_like(gray_all)
    Image.fromarray(gray_mask).save(proc_dir / "depth_gray_masked.png")

    Image.fromarray(colorize_depth(pred)).save(proc_dir / "depth_color_full.png")
    Image.fromarray(colorize_depth(pred, mask)).save(proc_dir / "depth_color_masked.png")

    depth_metric_png = res_dir / "depth_metric_mm.png"
    save_depth_png_meters(pred, depth_metric_png, scale=1000.0)

    scaffold_dir = res_dir / "metric_scaffold"
    scaffold_summary = build_from_depth(
        depth_path=depth_metric_png,
        image_path=args.image,
        mask_path=args.mask,
        intrinsics_path=args.intrinsics,
        output_dir=scaffold_dir,
        mask_threshold=args.mask_threshold,
        depth_scale=0.001,
    )
    scaffold_summary_path = scaffold_dir / "summary.json"
    scaffold_summary_path.write_text(json.dumps(scaffold_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    demo_summary = {
        "model_type": "apple_depth_pro",
        "checkpoint": str(args.checkpoint),
        "image": str(args.image),
        "mask": str(args.mask),
        "intrinsics": str(args.intrinsics),
        "predicted_focallength_px": focallength_px,
        "raw_depth_stats_m": {
            "min": raw_min,
            "max": raw_max,
            "mean": raw_mean,
        },
        "masked_depth_stats_m": {
            "min": masked_min,
            "max": masked_max,
            "mean": masked_mean,
        },
        "outputs": {
            "process_dir": str(proc_dir),
            "results_dir": str(res_dir),
            "depth_metric_mm_png": str(depth_metric_png),
            "metric_scaffold_summary": str(scaffold_summary_path),
        },
    }
    (args.output_dir / "demo_summary.json").write_text(json.dumps(demo_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    append_log(
        args.log_file,
        (
            f"- Built DepthPro metric-depth demo bundle at `{args.output_dir}`.",
            f"  - Predicted focal length: {focallength_px:.2f} px",
            f"  - Raw depth range: {raw_min:.4f}m to {raw_max:.4f}m",
            f"  - Masked food depth range: {masked_min:.4f}m to {masked_max:.4f}m",
        ),
    )
    print(f"DepthPro demo bundle ready: {args.output_dir}")


if __name__ == "__main__":
    main()
