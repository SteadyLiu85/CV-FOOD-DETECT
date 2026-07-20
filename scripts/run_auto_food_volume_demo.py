import argparse
import json
import subprocess
import sys
import time
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
    parser.add_argument("--auto-mask-backend", choices=["auto", "grounded_sam", "hf_groundingdino", "opencv"], default="auto")
    parser.add_argument("--hf-detector-model", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--volume-mode", choices=["support_plane", "scene_scale"], default="support_plane")
    parser.add_argument("--skip-volume", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("reports/noref_integration_log_20260511.md"))
    return parser.parse_args()


def auto_segment_food(image_path: Path, output_mask_path: Path, args: argparse.Namespace) -> Path:
    from auto_food_mask import generate_auto_mask

    result = generate_auto_mask(image_path, output_mask_path, args)
    metadata_path = output_mask_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_mask_path


def run_subprocess(cmd, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    pipeline_started = time.perf_counter()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = args.output_dir / "00_inputs"
    depth_dir = args.output_dir / "01_depthpro"
    volume_dir = args.output_dir / "02_volume"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    mask_path = args.mask
    timings = {
        "auto_mask_seconds": 0.0,
        "depthpro_seconds": 0.0,
        "volume_seconds": 0.0,
    }
    if mask_path is None:
        mask_path = inputs_dir / f"{args.image.stem}_auto_mask.png"
        step_started = time.perf_counter()
        auto_segment_food(args.image, mask_path, args)
        timings["auto_mask_seconds"] = time.perf_counter() - step_started
        auto_mask_metadata_path = mask_path.with_suffix(".metadata.json")
        auto_mask_info = json.loads(auto_mask_metadata_path.read_text(encoding="utf-8"))
    else:
        auto_mask_info = {
            "backend": "user_provided",
            "status": "ok",
            "mask": str(mask_path),
        }

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
    step_started = time.perf_counter()
    run_subprocess(depth_cmd, ROOT)
    timings["depthpro_seconds"] = time.perf_counter() - step_started

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
        step_started = time.perf_counter()
        run_subprocess(volume_cmd, ROOT)
        timings["volume_seconds"] = time.perf_counter() - step_started
        volume_summary = volume_dir / "metric_volume_calorie_summary.json"

    timings["total_seconds"] = time.perf_counter() - pipeline_started

    summary = {
        "image": str(args.image),
        "mask": str(mask_path),
        "auto_mask": auto_mask_info,
        "intrinsics": str(args.intrinsics) if args.intrinsics else None,
        "phone_model": args.phone_model,
        "phone_profiles": str(args.phone_profiles) if args.phone_profiles else None,
        "food": args.food,
        "depth_output_dir": str(depth_dir),
        "metric_summary": str(metric_summary),
        "volume_output_dir": str(volume_dir) if volume_summary else None,
        "volume_summary": str(volume_summary) if volume_summary else None,
        "timings_seconds": timings,
    }
    (args.output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
