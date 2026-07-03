import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_ascii_ply_vertices(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        vertex_count = 0
        for line in f:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        return np.loadtxt(f, max_rows=vertex_count, usecols=(0, 1, 2), dtype=np.float64)


def write_ascii_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
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


def find_mask(mask_dir: Path, stem: str) -> Path:
    candidates = [
        mask_dir / f"{stem}_segmented_mask.png",
        mask_dir / f"{stem}_segmented_mask.jpg",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Mask not found for stem={stem} in {mask_dir}")


def load_mask(mask_path: Path, size: Tuple[int, int], threshold: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return np.array(mask) > threshold


def world_to_camera_points(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    points_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (w2c @ points_h.T).T[:, :3]


def project_points_with_mode(
    cam_points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    perm: Tuple[int, int, int],
    signs: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam = np.stack([signs[i] * cam_points[:, perm[i]] for i in range(3)], axis=1)
    z = cam[:, 2]
    front = z > 1e-6
    front_indices = np.where(front)[0]
    cam = cam[front]
    z = z[front]
    if len(cam) == 0:
        empty_i = np.empty((0,), dtype=np.int32)
        empty_f = np.empty((0,), dtype=np.float64)
        empty_l = np.empty((0,), dtype=np.int64)
        return empty_i, empty_i, empty_f, empty_l
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    selected_indices = front_indices[inside]
    u = np.round(u[inside]).astype(np.int32)
    v = np.round(v[inside]).astype(np.int32)
    z = z[inside]
    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    return u, v, z, selected_indices


def visible_unique_pixels(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    width: int,
) -> np.ndarray:
    if len(u) == 0:
        return np.empty((0,), dtype=np.int64)
    lin = v.astype(np.int64) * width + u.astype(np.int64)
    order = np.argsort(z)
    _, first_idx = np.unique(lin[order], return_index=True)
    return order[first_idx]


def auto_search_projection_mode(
    points: np.ndarray,
    transforms: Dict,
    image_dir: Path,
    mask_dir: Path,
    probe_frames: List[int],
    threshold: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Dict]:
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    first_stem = Path(transforms["frames"][0]["file_path"]).stem
    image_path = image_dir / f"{first_stem}.jpg"
    if not image_path.exists():
        image_path = image_dir / f"{first_stem}.png"
    width, height = Image.open(image_path).size

    candidate_modes = [
        ((1, 0, 2), (-1, -1, -1)),
        ((1, 0, 2), (1, -1, -1)),
        ((0, 1, 2), (-1, 1, -1)),
        ((0, 1, 2), (1, -1, -1)),
    ]
    best_score = -1.0
    best_mode = candidate_modes[0]
    best_debug: Dict = {}

    for perm, signs in candidate_modes:
        total_hits = 0
        total_visible = 0
        per_frame = []
        for frame_idx in probe_frames:
            frame = transforms["frames"][frame_idx]
            stem = Path(frame["file_path"]).stem
            mask = load_mask(find_mask(mask_dir, stem), (width, height), threshold)
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            cam_points = world_to_camera_points(points, c2w)
            u, v, z, _ = project_points_with_mode(
                cam_points, fx, fy, cx, cy, width, height, perm, signs
            )
            keep = visible_unique_pixels(u, v, z, width)
            if len(keep) == 0:
                per_frame.append((frame_idx, stem, 0, 0, 0.0))
                continue
            hits = int(mask[v[keep], u[keep]].sum())
            visible = int(len(keep))
            ratio = hits / visible if visible else 0.0
            total_hits += hits
            total_visible += visible
            per_frame.append((frame_idx, stem, hits, visible, ratio))
        score = total_hits / total_visible if total_visible else 0.0
        if score > best_score:
            best_score = score
            best_mode = (perm, signs)
            best_debug = {
                "score": score,
                "total_hits": total_hits,
                "total_visible": total_visible,
                "per_frame": per_frame,
            }
    return best_mode[0], best_mode[1], best_debug


def build_from_scene_ply(
    scene_ply: Path,
    transforms_path: Path,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    frame_index: Optional[int],
    mask_threshold: int,
) -> Dict:
    points = read_ascii_ply_vertices(scene_ply)
    transforms = load_json(transforms_path)
    probe_frames = sorted(
        set(
            idx
            for idx in [
                0,
                len(transforms["frames"]) // 4,
                len(transforms["frames"]) // 2,
                (3 * len(transforms["frames"])) // 4,
                len(transforms["frames"]) - 1,
            ]
            if 0 <= idx < len(transforms["frames"])
        )
    )
    perm, signs, search_debug = auto_search_projection_mode(
        points, transforms, image_dir, mask_dir, probe_frames, mask_threshold
    )

    if frame_index is None:
        ranked = sorted(search_debug["per_frame"], key=lambda x: x[4], reverse=True)
        frame_index = ranked[0][0]

    frame = transforms["frames"][frame_index]
    stem = Path(frame["file_path"]).stem
    image_path = image_dir / f"{stem}.jpg"
    if not image_path.exists():
        image_path = image_dir / f"{stem}.png"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask_path = find_mask(mask_dir, stem)
    mask = load_mask(mask_path, (width, height), mask_threshold)

    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    cam_points = world_to_camera_points(points, c2w)
    u, v, z, selected_indices = project_points_with_mode(
        cam_points, fx, fy, cx, cy, width, height, perm, signs
    )
    keep = visible_unique_pixels(u, v, z, width)

    u_keep = u[keep]
    v_keep = v[keep]
    z_keep = z[keep]
    points_keep = points[selected_indices][keep]
    in_mask = mask[v_keep, u_keep]
    food_points = points_keep[in_mask]

    if len(food_points) == 0:
        raise RuntimeError("No scaffold points landed inside the selected mask.")

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    step_all = max(1, len(u_keep) // 2500)
    for x, y in zip(u_keep[::step_all], v_keep[::step_all]):
        draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=(0, 128, 255))
    if int(in_mask.sum()) > 0:
        step_sel = max(1, int(in_mask.sum()) // 1200)
        for x, y in zip(u_keep[in_mask][::step_sel], v_keep[in_mask][::step_sel]):
            draw.ellipse((int(x) - 2, int(y) - 2, int(x) + 2, int(y) + 2), fill=(255, 64, 64))

    masked_depth = np.zeros((height, width), dtype=np.float32)
    masked_depth[v_keep[in_mask], u_keep[in_mask]] = z_keep[in_mask]
    nonzero = masked_depth > 0
    if np.any(nonzero):
        zmin = float(masked_depth[nonzero].min())
        zmax = float(masked_depth[nonzero].max())
        norm = np.zeros_like(masked_depth, dtype=np.uint8)
        if zmax > zmin:
            scaled = (masked_depth[nonzero] - zmin) / (zmax - zmin)
            norm_vals = np.clip(255.0 * (1.0 - scaled), 0, 255).astype(np.uint8)
            norm[nonzero] = norm_vals
        else:
            norm[nonzero] = 255
    else:
        zmin = 0.0
        zmax = 0.0
        norm = np.zeros_like(masked_depth, dtype=np.uint8)

    scaffold_path = output_dir / "scaffold_scene_points.ply"
    write_ascii_ply(scaffold_path, food_points)
    overlay_path = output_dir / "overlay_projection.png"
    mask_path_out = output_dir / "mask_used.png"
    depth_path = output_dir / "masked_depth.png"
    overlay.save(overlay_path)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path_out)
    Image.fromarray(norm, mode="L").save(depth_path)

    mins = food_points.min(axis=0)
    maxs = food_points.max(axis=0)
    summary = {
        "mode": "scene_ply",
        "scene_ply": str(scene_ply),
        "transforms": str(transforms_path),
        "image": str(image_path),
        "mask": str(mask_path),
        "frame_index": int(frame_index),
        "frame_stem": stem,
        "projection_mode": {
            "perm": list(perm),
            "signs": list(signs),
            "auto_search": search_debug,
        },
        "counts": {
            "scene_vertices": int(len(points)),
            "visible_pixels": int(len(u_keep)),
            "food_scaffold_points": int(len(food_points)),
        },
        "bbox_world": {
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "extent": (maxs - mins).tolist(),
        },
        "masked_depth_range": {
            "min": zmin,
            "max": zmax,
        },
        "outputs": {
            "overlay_projection": str(overlay_path),
            "mask_used": str(mask_path_out),
            "masked_depth": str(depth_path),
            "scaffold_ply": str(scaffold_path),
        },
    }
    return summary


def build_from_depth(
    depth_path: Path,
    image_path: Path,
    mask_path: Path,
    intrinsics_path: Path,
    output_dir: Path,
    mask_threshold: int,
    depth_scale: float,
) -> Dict:
    intrinsics = load_json(intrinsics_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = load_mask(mask_path, (width, height), mask_threshold)

    depth_img = Image.open(depth_path)
    depth = np.array(depth_img, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape != (height, width):
        depth = np.array(depth_img.resize((width, height), Image.BILINEAR), dtype=np.float32)
    depth = depth * depth_scale

    fx = float(intrinsics["fl_x"])
    fy = float(intrinsics["fl_y"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    ys, xs = np.where(mask & (depth > 0))
    zs = depth[ys, xs]
    xs3 = (xs - cx) * zs / fx
    ys3 = (ys - cy) * zs / fy
    points = np.stack([xs3, ys3, zs], axis=1)

    if len(points) == 0:
        raise RuntimeError("Depth scaffold is empty after applying mask and positive-depth filter.")

    scaffold_path = output_dir / "scaffold_depth_points.ply"
    write_ascii_ply(scaffold_path, points)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    step = max(1, len(xs) // 2000)
    for x, y in zip(xs[::step], ys[::step]):
        draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=(255, 64, 64))
    overlay_path = output_dir / "overlay_projection.png"
    overlay.save(overlay_path)

    masked_depth = np.where(mask, depth, 0.0)
    nonzero = masked_depth > 0
    if np.any(nonzero):
        dmin = float(masked_depth[nonzero].min())
        dmax = float(masked_depth[nonzero].max())
        vis = np.zeros_like(masked_depth, dtype=np.uint8)
        if dmax > dmin:
            scaled = (masked_depth[nonzero] - dmin) / (dmax - dmin)
            vis[nonzero] = np.clip(255.0 * (1.0 - scaled), 0, 255).astype(np.uint8)
        else:
            vis[nonzero] = 255
    else:
        dmin = 0.0
        dmax = 0.0
        vis = np.zeros_like(masked_depth, dtype=np.uint8)
    depth_vis_path = output_dir / "masked_depth.png"
    Image.fromarray(vis, mode="L").save(depth_vis_path)
    mask_out = output_dir / "mask_used.png"
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_out)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        "mode": "depth",
        "depth": str(depth_path),
        "image": str(image_path),
        "mask": str(mask_path),
        "intrinsics": str(intrinsics_path),
        "counts": {
            "food_scaffold_points": int(len(points)),
        },
        "bbox_camera": {
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "extent": (maxs - mins).tolist(),
        },
        "masked_depth_range": {
            "min": dmin,
            "max": dmax,
        },
        "outputs": {
            "overlay_projection": str(overlay_path),
            "mask_used": str(mask_out),
            "masked_depth": str(depth_vis_path),
            "scaffold_ply": str(scaffold_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-reference food scaffold from either scene PLY or depth."
    )
    parser.add_argument(
        "--mode",
        choices=["scene_ply", "depth"],
        required=True,
        help="scene_ply reuses reconstructed scene points; depth consumes an external depth map.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=int, default=127)

    parser.add_argument("--scene-ply", type=Path)
    parser.add_argument("--transforms", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--masks-dir", type=Path)
    parser.add_argument("--frame-index", type=int)

    parser.add_argument("--depth", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiply raw depth values by this factor before back-projection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "scene_ply":
        required = [args.scene_ply, args.transforms, args.images_dir, args.masks_dir]
        if any(x is None for x in required):
            raise ValueError("scene_ply mode requires --scene-ply --transforms --images-dir --masks-dir")
        summary = build_from_scene_ply(
            scene_ply=args.scene_ply,
            transforms_path=args.transforms,
            image_dir=args.images_dir,
            mask_dir=args.masks_dir,
            output_dir=args.output_dir,
            frame_index=args.frame_index,
            mask_threshold=args.mask_threshold,
        )
    else:
        required = [args.depth, args.image, args.mask, args.intrinsics]
        if any(x is None for x in required):
            raise ValueError("depth mode requires --depth --image --mask --intrinsics")
        summary = build_from_depth(
            depth_path=args.depth,
            image_path=args.image,
            mask_path=args.mask,
            intrinsics_path=args.intrinsics,
            output_dir=args.output_dir,
            mask_threshold=args.mask_threshold,
            depth_scale=args.depth_scale,
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built no-reference scaffold bundle: {summary_path}")


if __name__ == "__main__":
    main()
