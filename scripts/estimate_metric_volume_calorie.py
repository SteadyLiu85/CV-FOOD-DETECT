import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageColor
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull

from build_noref_scaffold import (
    load_mask,
    load_json,
    project_points_with_mode,
    visible_unique_pixels,
    world_to_camera_points,
)


def append_log(log_path: Path, lines: Tuple[str, ...]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def resolve_repo_relative(base_file: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    repo_root = base_file.resolve().parents[2]
    return (repo_root / p).resolve()


def read_ascii_ply_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        vertex_count = 0
        face_count = 0
        for line in f:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face "):
                face_count = int(line.split()[-1])
            elif line.strip() == "end_header":
                break

        vertices = np.loadtxt(f, max_rows=vertex_count, usecols=(0, 1, 2), dtype=np.float64)
        if vertices.ndim == 1:
            vertices = vertices[None, :]

        faces = []
        for _ in range(face_count):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if not parts:
                continue
            n = int(parts[0])
            idx = [int(x) for x in parts[1 : 1 + n]]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) > 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return vertices, np.array(faces, dtype=np.int64)


def read_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            idx = []
            for token in line.split()[1:]:
                idx.append(int(token.split("/")[0]) - 1)
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) > 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def write_ascii_ply_points(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(points, colors):
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )


def mesh_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    if len(faces) == 0:
        return 0.0
    tris = vertices[faces]
    return float(
        np.sum(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2]))) / 6.0
    )


def mesh_is_watertight(faces: np.ndarray) -> bool:
    if len(faces) == 0:
        return False
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> Dict:
    signed_volume_m3 = abs(mesh_signed_volume(vertices, faces))
    hull_volume_m3 = float(ConvexHull(vertices).volume)
    watertight = mesh_is_watertight(faces)
    chosen_source = "mesh_signed_volume" if watertight and signed_volume_m3 > 0 else "convex_hull"
    chosen_volume_m3 = signed_volume_m3 if chosen_source == "mesh_signed_volume" else hull_volume_m3
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "watertight": watertight,
        "signed_volume_m3": signed_volume_m3,
        "convex_hull_volume_m3": hull_volume_m3,
        "chosen_volume_m3": chosen_volume_m3,
        "chosen_volume_source": chosen_source,
        "bbox_extent_m": extents.tolist(),
    }


def load_depth_png_meters(path: Path, scale: float) -> np.ndarray:
    depth = np.array(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth * scale


def backproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x3 = (xs - cx) * zs / fx
    y3 = (ys - cy) * zs / fy
    return np.stack([x3, y3, zs], axis=1)


def robust_scale_ratios(ratios: np.ndarray) -> Dict:
    p10, p50, p90 = np.percentile(ratios, [10.0, 50.0, 90.0])
    iqr_lo, iqr_hi = np.percentile(ratios, [25.0, 75.0])
    keep = (ratios >= iqr_lo - 1.5 * (iqr_hi - iqr_lo)) & (ratios <= iqr_hi + 1.5 * (iqr_hi - iqr_lo))
    trimmed = ratios[keep] if np.any(keep) else ratios
    return {
        "count": int(len(ratios)),
        "count_trimmed": int(len(trimmed)),
        "median_m_per_unit": float(np.median(trimmed)),
        "mean_m_per_unit": float(trimmed.mean()),
        "std_m_per_unit": float(trimmed.std()),
        "p10_m_per_unit": float(p10),
        "p50_m_per_unit": float(p50),
        "p90_m_per_unit": float(p90),
    }


def estimate_scene_scale_from_metric(
    scene_summary_path: Path,
    metric_summary_path: Path,
    metric_depth_scale: float,
) -> Dict:
    scene_summary = load_json(scene_summary_path)
    metric_summary = load_json(metric_summary_path)

    scene_ply = resolve_repo_relative(scene_summary_path, scene_summary["scene_ply"])
    transforms_path = resolve_repo_relative(scene_summary_path, scene_summary["transforms"])
    image_path = resolve_repo_relative(scene_summary_path, scene_summary["image"])
    mask_path = resolve_repo_relative(scene_summary_path, scene_summary["mask"])
    metric_depth_path = resolve_repo_relative(metric_summary_path, metric_summary["depth"])

    transforms = load_json(transforms_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = load_mask(mask_path, (width, height), threshold=127)
    metric_depth = load_depth_png_meters(metric_depth_path, metric_depth_scale)
    if metric_depth.shape != (height, width):
        metric_depth = np.array(
            Image.fromarray(metric_depth).resize((width, height), Image.BILINEAR),
            dtype=np.float32,
        )

    vertices, _ = read_ascii_ply_mesh(scene_ply)
    frame_index = int(scene_summary["frame_index"])
    perm = tuple(int(x) for x in scene_summary["projection_mode"]["perm"])
    signs = tuple(int(x) for x in scene_summary["projection_mode"]["signs"])
    frame = transforms["frames"][frame_index]
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)

    cam_points = world_to_camera_points(vertices, c2w)
    u, v, z_scene, selected_indices = project_points_with_mode(
        cam_points,
        fx=float(transforms["fl_x"]),
        fy=float(transforms["fl_y"]),
        cx=float(transforms["cx"]),
        cy=float(transforms["cy"]),
        width=width,
        height=height,
        perm=perm,
        signs=signs,
    )
    keep = visible_unique_pixels(u, v, z_scene, width)
    u_keep = u[keep]
    v_keep = v[keep]
    z_keep = z_scene[keep]

    scene_depth_map = np.zeros((height, width), dtype=np.float32)
    scene_depth_map[v_keep, u_keep] = z_keep.astype(np.float32)
    overlap = mask & (scene_depth_map > 0) & (metric_depth > 0)
    if not np.any(overlap):
        raise RuntimeError("No overlapping valid pixels between scene scaffold and metric depth.")

    ratios = metric_depth[overlap] / scene_depth_map[overlap]
    scale_stats = robust_scale_ratios(ratios.astype(np.float64))

    vis = np.array(image, dtype=np.uint8)
    green = np.array(ImageColor.getrgb("#58d68d"), dtype=np.uint8)
    red = np.array(ImageColor.getrgb("#ff5c5c"), dtype=np.uint8)
    vis[mask] = (0.65 * vis[mask] + 0.35 * green).astype(np.uint8)
    vis[overlap] = (0.55 * vis[overlap] + 0.45 * red).astype(np.uint8)

    scene_overlap_points = vertices[selected_indices][keep][overlap[v_keep, u_keep]]
    return {
        "scene_ply": str(scene_ply),
        "transforms": str(transforms_path),
        "metric_depth": str(metric_depth_path),
        "scene_depth_map": scene_depth_map,
        "metric_depth_map": metric_depth,
        "mask": mask,
        "overlay_image": vis,
        "scale_stats": scale_stats,
        "scene_overlap_points": scene_overlap_points,
        "overlap_count": int(np.count_nonzero(overlap)),
    }


def fit_plane_ransac(points: np.ndarray, rng: np.random.Generator, threshold: float, iterations: int) -> Tuple[np.ndarray, float, np.ndarray]:
    if len(points) < 3:
        raise RuntimeError("Need at least three points to fit a plane.")

    best_inliers = None
    best_count = -1
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, sample[0]))
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        raise RuntimeError("RANSAC plane fit failed.")

    pts = points[best_inliers]
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(np.dot(normal, center))
    return normal, d, best_inliers


def pixel_corner_base_area(
    xs: np.ndarray,
    ys: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    normal: np.ndarray,
    d: float,
) -> Tuple[np.ndarray, np.ndarray]:
    uc = np.stack([xs - 0.5, xs + 0.5, xs + 0.5, xs - 0.5], axis=1)
    vc = np.stack([ys - 0.5, ys - 0.5, ys + 0.5, ys + 0.5], axis=1)
    rx = (uc - cx) / fx
    ry = (vc - cy) / fy
    denom = normal[0] * rx + normal[1] * ry + normal[2]
    valid = np.all(np.abs(denom) > 1e-8, axis=1)
    t = np.zeros_like(denom)
    t[valid] = (-d) / denom[valid]

    base = np.stack([rx * t, ry * t, t], axis=2)
    tri1 = np.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0])
    tri2 = np.cross(base[:, 2] - base[:, 0], base[:, 3] - base[:, 0])
    area = 0.5 * np.linalg.norm(tri1, axis=1) + 0.5 * np.linalg.norm(tri2, axis=1)
    return area, valid


def estimate_support_plane_volume(metric_summary_path: Path, metric_depth_scale: float) -> Dict:
    metric_summary = load_json(metric_summary_path)
    depth_path = resolve_repo_relative(metric_summary_path, metric_summary["depth"])
    image_path = resolve_repo_relative(metric_summary_path, metric_summary["image"])
    mask_path = resolve_repo_relative(metric_summary_path, metric_summary["mask"])
    intrinsics_path = resolve_repo_relative(metric_summary_path, metric_summary["intrinsics"])

    intrinsics = load_json(intrinsics_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = load_mask(mask_path, (width, height), threshold=127)
    depth = load_depth_png_meters(depth_path, metric_depth_scale)

    ys_mask, xs_mask = np.where(mask & (depth > 0))
    zs_mask = depth[ys_mask, xs_mask]
    fx = float(intrinsics["fl_x"])
    fy = float(intrinsics["fl_y"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    food_points = backproject_pixels(xs_mask.astype(np.float64), ys_mask.astype(np.float64), zs_mask.astype(np.float64), fx, fy, cx, cy)

    x0, x1 = int(xs_mask.min()), int(xs_mask.max())
    y0, y1 = int(ys_mask.min()), int(ys_mask.max())
    margin = int(max(x1 - x0 + 1, y1 - y0 + 1) * 0.20)
    dilated = binary_dilation(mask, iterations=max(8, margin // 6))
    ring = dilated & (~mask)
    ring_roi = np.zeros_like(ring)
    ring_roi[max(0, y0 - margin) : min(height, y1 + margin + 1), max(0, x0 - margin) : min(width, x1 + margin + 1)] = True
    ring = ring & ring_roi & (depth > 0)

    ys_ring, xs_ring = np.where(ring)
    zs_ring = depth[ys_ring, xs_ring]
    ring_points = backproject_pixels(xs_ring.astype(np.float64), ys_ring.astype(np.float64), zs_ring.astype(np.float64), fx, fy, cx, cy)
    if len(ring_points) < 200:
        raise RuntimeError("Not enough support-plane points around the object.")

    rng = np.random.default_rng(42)
    normal, d, inliers = fit_plane_ransac(ring_points, rng=rng, threshold=0.006, iterations=400)
    signed_food = food_points @ normal + d
    if np.median(signed_food) < 0:
        normal = -normal
        d = -d
        signed_food = -signed_food

    base_area, valid_area = pixel_corner_base_area(
        xs_mask.astype(np.float64),
        ys_mask.astype(np.float64),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        normal=normal,
        d=d,
    )
    valid_height = signed_food > 0
    valid = valid_area & valid_height
    if not np.any(valid):
        raise RuntimeError("Support-plane integration produced no valid pixels.")

    volume_m3 = float(np.sum(base_area[valid] * signed_food[valid]))
    base_points = food_points - signed_food[:, None] * normal[None, :]

    overlay = np.array(image, dtype=np.uint8)
    overlay[ring] = (0.65 * overlay[ring] + 0.35 * np.array([80, 160, 255], dtype=np.uint8)).astype(np.uint8)
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 90, 90], dtype=np.uint8)).astype(np.uint8)

    height_map = np.zeros((height, width), dtype=np.float32)
    height_map[ys_mask, xs_mask] = signed_food.astype(np.float32)
    positive = height_map > 0
    height_vis = np.zeros((height, width), dtype=np.uint8)
    if np.any(positive):
        hmin = float(height_map[positive].min())
        hmax = float(height_map[positive].max())
        scaled = np.zeros_like(height_map, dtype=np.float32)
        if hmax > hmin:
            scaled[positive] = (height_map[positive] - hmin) / (hmax - hmin)
        height_vis[positive] = np.clip(255 * scaled[positive], 0, 255).astype(np.uint8)
    else:
        hmin = 0.0
        hmax = 0.0

    return {
        "image": str(image_path),
        "mask": str(mask_path),
        "intrinsics": str(intrinsics_path),
        "depth": str(depth_path),
        "overlay_image": overlay,
        "height_vis": height_vis,
        "food_points": food_points,
        "base_points": base_points,
        "ring_mask": ring,
        "plane_normal": normal,
        "plane_d": d,
        "plane_inlier_ratio": float(np.mean(inliers)),
        "volume_m3": volume_m3,
        "height_stats_m": {
            "min": hmin,
            "max": hmax,
            "mean": float(signed_food[valid].mean()),
        },
        "valid_pixel_count": int(np.count_nonzero(valid)),
    }


def attach_nutrition(result: Dict, food: str, priors_path: Path) -> Dict:
    priors = load_json(priors_path)
    if food not in priors:
        raise KeyError(f"Food prior not found: {food}")
    prior = priors[food]
    volume_ml = float(result["chosen_volume_m3"]) * 1_000_000.0
    density = float(prior["density_g_per_ml"]["mean"])
    mass_g = volume_ml * density
    kcal_per_100g = float(prior["kcal_per_100g"]["mean"])
    kcal = mass_g * kcal_per_100g / 100.0
    return {
        "food": food,
        "display_name": prior.get("display_name", food),
        "prior": prior,
        "physical_estimation": {
            "estimated_volume_ml": volume_ml,
            "estimated_mass_g": mass_g,
            "estimated_kcal": kcal,
            "density_g_per_ml": density,
            "kcal_per_100g": kcal_per_100g,
        },
    }


def compare_gt(gt_mesh_path: Optional[Path]) -> Optional[Dict]:
    if gt_mesh_path is None:
        return None
    vertices, faces = read_obj_mesh(gt_mesh_path)
    gt_metrics = mesh_metrics(vertices, faces)
    gt_volume_ml = gt_metrics["chosen_volume_m3"] * 1_000_000.0
    gt_metrics["chosen_volume_ml"] = gt_volume_ml
    gt_metrics["mesh_path"] = str(gt_mesh_path)
    return gt_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close the metric-depth branch to volume, mass, and calorie estimates.")
    parser.add_argument("--mode", choices=["scene_scale", "support_plane"], required=True)
    parser.add_argument("--food", type=str, required=True)
    parser.add_argument("--priors", type=Path, default=Path("assets/noref_food_priors.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric-summary", type=Path, required=True, help="summary.json produced by build_noref_scaffold.py in depth mode")
    parser.add_argument("--scene-summary", type=Path, help="summary.json produced by build_noref_scaffold.py in scene_ply mode")
    parser.add_argument("--gt-mesh", type=Path, help="Optional ground-truth mesh for validation")
    parser.add_argument("--metric-depth-scale", type=float, default=0.001)
    parser.add_argument("--log-file", type=Path, default=Path("reports/noref_integration_log_20260511.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "scene_scale":
        if args.scene_summary is None:
            raise ValueError("scene_scale mode requires --scene-summary")
        scale_data = estimate_scene_scale_from_metric(
            scene_summary_path=args.scene_summary,
            metric_summary_path=args.metric_summary,
            metric_depth_scale=args.metric_depth_scale,
        )
        scene_summary = load_json(args.scene_summary)
        scene_ply = resolve_repo_relative(args.scene_summary, scene_summary["scene_ply"])
        vertices, faces = read_ascii_ply_mesh(scene_ply)
        scale_m_per_unit = scale_data["scale_stats"]["median_m_per_unit"]
        scaled_vertices = vertices * scale_m_per_unit
        volume_info = mesh_metrics(scaled_vertices, faces)
        gt_info = compare_gt(args.gt_mesh)
        nutrition = attach_nutrition(volume_info, food=args.food, priors_path=args.priors)

        Image.fromarray(scale_data["overlay_image"]).save(args.output_dir / "overlap_overlay.png")
        write_ascii_ply_points(args.output_dir / "scene_overlap_points_metric_scaled.ply", scale_data["scene_overlap_points"] * scale_m_per_unit)

        result = {
            "mode": args.mode,
            "scene_summary": str(args.scene_summary),
            "metric_summary": str(args.metric_summary),
            "alignment": {
                "overlap_count": scale_data["overlap_count"],
                "scale_stats": scale_data["scale_stats"],
            },
            "mesh_metrics": volume_info,
            **nutrition,
            "ground_truth": gt_info,
            "outputs": {
                "overlap_overlay": str(args.output_dir / "overlap_overlay.png"),
                "scene_overlap_points_metric_scaled": str(args.output_dir / "scene_overlap_points_metric_scaled.ply"),
            },
            "limitations": [
                "This branch assumes the metric depth and reconstructed mesh correspond to the same food mask.",
                "The scale is estimated from overlapping visible pixels, so reconstruction holes can bias the scale.",
            ],
        }
        if gt_info is not None and gt_info["chosen_volume_ml"] > 0:
            est_ml = nutrition["physical_estimation"]["estimated_volume_ml"]
            result["ground_truth"]["volume_mape_percent"] = 100.0 * abs(est_ml - gt_info["chosen_volume_ml"]) / gt_info["chosen_volume_ml"]

        append_log(
            args.log_file,
            (
                f"- Closed the metric-depth -> mesh-scale -> calorie branch for `{args.food}`.",
                f"  - Mode: scene_scale",
                f"  - Estimated scale: {scale_m_per_unit:.6f} m/unit",
                f"  - Estimated volume: {nutrition['physical_estimation']['estimated_volume_ml']:.2f} mL",
                f"  - Estimated calories: {nutrition['physical_estimation']['estimated_kcal']:.2f} kcal",
            ),
        )
    else:
        plane_data = estimate_support_plane_volume(
            metric_summary_path=args.metric_summary,
            metric_depth_scale=args.metric_depth_scale,
        )
        volume_info = {
            "chosen_volume_m3": plane_data["volume_m3"],
            "chosen_volume_source": "support_plane_integration",
            "bbox_extent_m": (plane_data["food_points"].max(axis=0) - plane_data["food_points"].min(axis=0)).tolist(),
            "vertex_count": int(len(plane_data["food_points"])),
            "face_count": 0,
            "watertight": False,
            "signed_volume_m3": 0.0,
            "convex_hull_volume_m3": float(ConvexHull(plane_data["food_points"]).volume),
        }
        gt_info = compare_gt(args.gt_mesh)
        nutrition = attach_nutrition(volume_info, food=args.food, priors_path=args.priors)

        Image.fromarray(plane_data["overlay_image"]).save(args.output_dir / "support_plane_overlay.png")
        Image.fromarray(plane_data["height_vis"], mode="L").save(args.output_dir / "height_map.png")
        write_ascii_ply_points(
            args.output_dir / "support_plane_points.ply",
            np.concatenate([plane_data["food_points"], plane_data["base_points"]], axis=0),
            colors=np.concatenate(
                [
                    np.tile(np.array([[255, 80, 80]], dtype=np.uint8), (len(plane_data["food_points"]), 1)),
                    np.tile(np.array([[80, 160, 255]], dtype=np.uint8), (len(plane_data["base_points"]), 1)),
                ],
                axis=0,
            ),
        )

        result = {
            "mode": args.mode,
            "metric_summary": str(args.metric_summary),
            "support_plane": {
                "plane_normal": plane_data["plane_normal"].tolist(),
                "plane_d": float(plane_data["plane_d"]),
                "plane_inlier_ratio": plane_data["plane_inlier_ratio"],
                "valid_pixel_count": plane_data["valid_pixel_count"],
                "height_stats_m": plane_data["height_stats_m"],
            },
            "mesh_metrics": volume_info,
            **nutrition,
            "ground_truth": gt_info,
            "outputs": {
                "support_plane_overlay": str(args.output_dir / "support_plane_overlay.png"),
                "height_map": str(args.output_dir / "height_map.png"),
                "support_plane_points": str(args.output_dir / "support_plane_points.ply"),
            },
            "limitations": [
                "This branch treats the visible food surface as a height field over a locally fitted support plane.",
                "It is better suited to plate-supported foods than soups, mixed dishes, or objects with strong overhangs.",
            ],
        }
        if gt_info is not None and gt_info["chosen_volume_ml"] > 0:
            est_ml = nutrition["physical_estimation"]["estimated_volume_ml"]
            result["ground_truth"]["volume_mape_percent"] = 100.0 * abs(est_ml - gt_info["chosen_volume_ml"]) / gt_info["chosen_volume_ml"]

        append_log(
            args.log_file,
            (
                f"- Closed the metric-depth -> support-plane -> calorie branch for `{args.food}`.",
                f"  - Mode: support_plane",
                f"  - Estimated volume: {nutrition['physical_estimation']['estimated_volume_ml']:.2f} mL",
                f"  - Estimated calories: {nutrition['physical_estimation']['estimated_kcal']:.2f} kcal",
            ),
        )

    out_path = args.output_dir / "metric_volume_calorie_summary.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote metric volume/calorie summary: {out_path}")


if __name__ == "__main__":
    main()
