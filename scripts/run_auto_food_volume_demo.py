import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click food volume demo: auto mask -> auto intrinsics -> DepthPro -> support-plane volume."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to DepthPro checkpoint .pt file")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--food", type=str, help="Food class key used by estimate_metric_volume_calorie.py")
    parser.add_argument("--mask", type=Path, help="Optional existing food mask. If omitted, generate one automatically.")
    parser.add_argument("--intrinsics", type=Path, help="Optional existing intrinsics JSON.")
    parser.add_argument("--phone-model", type=str, help="Optional phone model used for profile matching.")
    parser.add_argument("--phone-profiles", type=Path, help="Optional JSON database keyed by phone model.")
    parser.add_argument("--prompt", type=str, default="food . dish . meal . fruit . rice . noodle . meat . bread . soup")
    parser.add_argument("--min-side", type=int, default=480)
    parser.add_argument("--dino-threshold", type=float, default=0.35)
    parser.add_argument("--dino-nms-threshold", type=float, default=0.8)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--volume-mode", choices=["support_plane", "scene_scale"], default="support_plane")
    parser.add_argument("--skip-volume", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("reports/noref_integration_log_20260511.md"))
    return parser.parse_args()


def auto_segment_food(image_path: Path, output_mask_path: Path, args: argparse.Namespace) -> Path:
    import torch
    from run_grounded_sam_to_voleta import (
        build_config,
        get_grounding_dino_model,
        load_image_rgb,
        save_binary_mask,
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
        mask, _segments_info = segment_with_text(
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
    save_binary_mask(mask_np, output_mask_path)
    return output_mask_path


def run_subprocess(cmd, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.output_dir / "00_inputs"
    depth_dir = args.output_dir / "01_depthpro"
    volume_dir = args.output_dir / "02_volume"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    mask_path = args.mask
    if mask_path is None:
        mask_path = inputs_dir / f"{args.image.stem}_auto_mask.png"
        auto_segment_food(args.image, mask_path, args)

    depth_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_depthpro_demo.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--image",
        str(args.image),
        "--mask",
        str(mask_path),
        "--output-dir",
        str(depth_dir),
        "--mask-threshold",
        str(args.mask_threshold),
        "--log-file",
        str(args.log_file),
    ]
    if args.intrinsics is not None:
        depth_cmd.extend(["--intrinsics", str(args.intrinsics)])
    if args.phone_model:
        depth_cmd.extend(["--phone-model", args.phone_model])
    if args.phone_profiles is not None:
        depth_cmd.extend(["--phone-profiles", str(args.phone_profiles)])
    run_subprocess(depth_cmd, ROOT)

    metric_summary = depth_dir / "02_results" / "metric_scaffold" / "summary.json"

    volume_summary = None
    if not args.skip_volume and args.food:
        volume_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "estimate_metric_volume_calorie.py"),
            "--mode",
            args.volume_mode,
            "--food",
            args.food,
            "--metric-summary",
            str(metric_summary),
            "--output-dir",
            str(volume_dir),
            "--log-file",
            str(args.log_file),
        ]
        run_subprocess(volume_cmd, ROOT)
        volume_summary = volume_dir / "metric_volume_calorie_summary.json"

    summary = {
        "image": str(args.image),
        "mask": str(mask_path),
        "intrinsics": str(args.intrinsics) if args.intrinsics else None,
        "phone_model": args.phone_model,
        "phone_profiles": str(args.phone_profiles) if args.phone_profiles else None,
        "food": args.food,
        "depth_output_dir": str(depth_dir),
        "metric_summary": str(metric_summary),
        "volume_output_dir": str(volume_dir) if volume_summary else None,
        "volume_summary": str(volume_summary) if volume_summary else None,
    }
    (args.output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
