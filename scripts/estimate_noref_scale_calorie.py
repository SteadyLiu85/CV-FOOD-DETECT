import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import ConvexHull


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_relative(base_file: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    repo_root = base_file.resolve().parents[2]
    return (repo_root / p).resolve()


def append_log(log_path: Path, lines: Tuple[str, ...]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def read_ascii_ply_vertices(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        vertex_count = 0
        for line in f:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        pts = np.loadtxt(f, max_rows=vertex_count, usecols=(0, 1, 2), dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    return pts


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def pca_extents(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh.T
    return np.sort(proj.max(axis=0) - proj.min(axis=0))[::-1]


def crop_full_scene(scene_points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin_ratio: float) -> np.ndarray:
    extent = bbox_max - bbox_min
    margin = margin_ratio * extent.max()
    lo = bbox_min - margin
    hi = bbox_max + margin
    keep = (
        (scene_points[:, 0] >= lo[0])
        & (scene_points[:, 0] <= hi[0])
        & (scene_points[:, 1] >= lo[1])
        & (scene_points[:, 1] <= hi[1])
        & (scene_points[:, 2] >= lo[2])
        & (scene_points[:, 2] <= hi[2])
    )
    return scene_points[keep]


def summarize_points(points: np.ndarray) -> Dict:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extents = bbox_max - bbox_min
    pca_dims = pca_extents(points)
    hull = ConvexHull(points)
    return {
        "count": int(len(points)),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": extents.tolist(),
        "pca_dims": pca_dims.tolist(),
        "raw_hull_volume": float(hull.volume),
    }


def estimate_scale_from_prior(raw_dims: np.ndarray, prior_dims_cm: np.ndarray) -> Dict:
    raw_sorted = np.sort(raw_dims)[::-1]
    prior_sorted = np.sort(prior_dims_cm)[::-1]
    per_axis = prior_sorted / np.maximum(raw_sorted, 1e-8)
    scale_cm_per_unit = float(np.median(per_axis))
    return {
        "raw_dims_sorted": raw_sorted.tolist(),
        "prior_dims_sorted_cm": prior_sorted.tolist(),
        "per_axis_scales_cm_per_unit": per_axis.tolist(),
        "chosen_scale_cm_per_unit": scale_cm_per_unit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate physical scale, volume, mass, and calories from a no-reference scaffold."
    )
    parser.add_argument("--summary", type=Path, required=True, help="summary.json produced by build_noref_scaffold.py")
    parser.add_argument("--food", type=str, required=True, help="Food class key used to select priors, e.g. strawberry")
    parser.add_argument("--priors", type=Path, default=Path("assets/noref_food_priors.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.12)
    parser.add_argument("--use-full-scene-crop", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("reports/noref_integration_log_20260511.md"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_json(args.summary)
    priors = load_json(args.priors)
    if args.food not in priors:
        raise KeyError(f"Food prior not found: {args.food}")
    prior = priors[args.food]

    scaffold_path = resolve_repo_relative(args.summary, summary["outputs"]["scaffold_ply"])
    scaffold_points = read_ascii_ply_vertices(scaffold_path)
    source_mode = "visible_scaffold"
    points = scaffold_points

    full_scene_crop_path = None
    if args.use_full_scene_crop and summary.get("scene_ply"):
        scene_ply = resolve_repo_relative(args.summary, summary["scene_ply"])
        scene_points = read_ascii_ply_vertices(scene_ply)
        bbox_min = np.array(summary["bbox_world"]["min"], dtype=np.float64)
        bbox_max = np.array(summary["bbox_world"]["max"], dtype=np.float64)
        cropped = crop_full_scene(scene_points, bbox_min, bbox_max, args.crop_margin_ratio)
        if len(cropped) >= len(scaffold_points):
            points = cropped
            source_mode = "full_scene_crop"
            full_scene_crop_path = args.output_dir / "scene_crop_points.ply"
            write_ascii_ply(full_scene_crop_path, cropped)

    geometry = summarize_points(points)
    prior_dims = np.array(prior["size_prior_cm"]["mean"], dtype=np.float64)
    scale = estimate_scale_from_prior(np.array(geometry["pca_dims"], dtype=np.float64), prior_dims)
    scale_cm_per_unit = scale["chosen_scale_cm_per_unit"]

    raw_volume = float(geometry["raw_hull_volume"])
    volume_ml = raw_volume * (scale_cm_per_unit ** 3)
    density = float(prior["density_g_per_ml"]["mean"])
    mass_g = volume_ml * density
    kcal_per_100g = float(prior["kcal_per_100g"]["mean"])
    kcal = mass_g * kcal_per_100g / 100.0

    result = {
        "food": args.food,
        "display_name": prior.get("display_name", args.food),
        "source_mode": source_mode,
        "input_summary": str(args.summary),
        "input_scaffold": str(scaffold_path),
        "prior_file": str(args.priors),
        "prior": prior,
        "geometry": geometry,
        "scale_estimation": scale,
        "physical_estimation": {
            "estimated_volume_ml": volume_ml,
            "estimated_mass_g": mass_g,
            "estimated_kcal": kcal,
            "density_g_per_ml": density,
            "kcal_per_100g": kcal_per_100g,
        },
        "outputs": {
            "scene_crop_points": str(full_scene_crop_path) if full_scene_crop_path else None
        },
        "limitations": [
            "This fallback estimates scale from semantic size priors, not metric depth.",
            "Suitable for single-item foods with relatively stable shape and size distributions.",
            "Not reliable for mixed dishes, soups, sauces, or heavily occluded foods."
        ]
    }

    out_path = args.output_dir / "scale_calorie_summary.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(
        args.log_file,
        (
            f"- Built no-reference scale/calorie estimate for `{args.food}`.",
            f"  - Source mode: {source_mode}",
            f"  - Estimated scale: {scale_cm_per_unit:.4f} cm/unit",
            f"  - Estimated volume: {volume_ml:.2f} mL",
            f"  - Estimated mass: {mass_g:.2f} g",
            f"  - Estimated calories: {kcal:.2f} kcal",
            "  - Method: semantic size prior fallback, pending metric-depth upgrade.",
        ),
    )
    print(f"Wrote scale/calorie summary: {out_path}")


if __name__ == "__main__":
    main()
