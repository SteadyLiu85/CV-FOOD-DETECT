import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import ExifTags, Image


FULL_FRAME_WIDTH_MM = 36.0
FULL_FRAME_HEIGHT_MM = 24.0


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rational_to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    num = getattr(value, "numerator", None)
    den = getattr(value, "denominator", None)
    if num is not None and den:
        return float(num) / float(den)
    return None


def normalize_phone_key(value: str) -> str:
    return " ".join((value or "").strip().upper().split())


def extract_exif_info(image_path: Path) -> Dict:
    with Image.open(image_path) as image:
        width, height = image.size
        exif_raw = image.getexif()

    exif_map = {}
    for key, value in exif_raw.items():
        exif_map[ExifTags.TAGS.get(key, str(key))] = value

    make = str(exif_map.get("Make", "")).strip()
    model = str(exif_map.get("Model", "")).strip()
    lens_model = str(exif_map.get("LensModel", "")).strip()
    focal_length_mm = rational_to_float(exif_map.get("FocalLength"))
    focal_35mm_eq = rational_to_float(exif_map.get("FocalLengthIn35mmFilm"))

    return {
        "width": width,
        "height": height,
        "make": make,
        "model": model,
        "lens_model": lens_model,
        "focal_length_mm": focal_length_mm,
        "focal_length_35mm_equiv_mm": focal_35mm_eq,
        "normalized_make_model": normalize_phone_key(f"{make} {model}"),
        "normalized_model": normalize_phone_key(model),
    }


def load_phone_profiles(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None or not path.exists():
        return {}
    raw = load_json(path)
    profiles = raw.get("profiles", raw)
    normalized = {}
    for key, value in profiles.items():
        normalized[normalize_phone_key(key)] = value
    return normalized


def find_matching_profile(
    profiles: Dict[str, Dict],
    phone_model: Optional[str],
    exif_info: Dict,
) -> Tuple[Optional[Dict], Optional[str]]:
    candidates = []
    if phone_model:
        candidates.append(normalize_phone_key(phone_model))
    if exif_info.get("normalized_make_model"):
        candidates.append(exif_info["normalized_make_model"])
    if exif_info.get("normalized_model"):
        candidates.append(exif_info["normalized_model"])

    for key in candidates:
        if key and key in profiles:
            return profiles[key], key
    return None, None


def base_intrinsics(width: int, height: int, fl_x: float, fl_y: float) -> Dict:
    return {
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float((width - 1) / 2.0),
        "cy": float((height - 1) / 2.0),
        "w": float(width),
        "h": float(height),
    }


def intrinsics_from_profile(width: int, height: int, profile: Dict) -> Optional[Dict]:
    if "fl_x" in profile and "fl_y" in profile:
        return base_intrinsics(width, height, float(profile["fl_x"]), float(profile["fl_y"]))

    if "focal_px" in profile:
        focal_px = float(profile["focal_px"])
        return base_intrinsics(width, height, focal_px, focal_px)

    if all(k in profile for k in ("focal_length_mm", "sensor_width_mm", "sensor_height_mm")):
        focal_length_mm = float(profile["focal_length_mm"])
        sensor_width_mm = float(profile["sensor_width_mm"])
        sensor_height_mm = float(profile["sensor_height_mm"])
        fl_x = width * focal_length_mm / sensor_width_mm
        fl_y = height * focal_length_mm / sensor_height_mm
        return base_intrinsics(width, height, fl_x, fl_y)

    if "focal_35mm_equiv_mm" in profile:
        focal_35 = float(profile["focal_35mm_equiv_mm"])
        fl_x = width * focal_35 / FULL_FRAME_WIDTH_MM
        fl_y = height * focal_35 / FULL_FRAME_HEIGHT_MM
        return base_intrinsics(width, height, fl_x, fl_y)

    if "hfov_deg" in profile:
        hfov_deg = float(profile["hfov_deg"])
        fl_x = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        return base_intrinsics(width, height, fl_x, fl_x)

    return None


def intrinsics_from_exif(width: int, height: int, exif_info: Dict, profile: Optional[Dict]) -> Optional[Dict]:
    if profile and all(k in profile for k in ("sensor_width_mm", "sensor_height_mm")) and exif_info.get("focal_length_mm") is not None:
        focal_length_mm = float(exif_info["focal_length_mm"])
        sensor_width_mm = float(profile["sensor_width_mm"])
        sensor_height_mm = float(profile["sensor_height_mm"])
        fl_x = width * focal_length_mm / sensor_width_mm
        fl_y = height * focal_length_mm / sensor_height_mm
        return base_intrinsics(width, height, fl_x, fl_y)

    focal_35 = exif_info.get("focal_length_35mm_equiv_mm")
    if focal_35 is not None and focal_35 > 0:
        fl_x = width * float(focal_35) / FULL_FRAME_WIDTH_MM
        fl_y = height * float(focal_35) / FULL_FRAME_HEIGHT_MM
        return base_intrinsics(width, height, fl_x, fl_y)

    return None


def intrinsics_from_focal_px(width: int, height: int, focal_px: float) -> Dict:
    return base_intrinsics(width, height, float(focal_px), float(focal_px))


def resolve_intrinsics_for_image(
    image_path: Path,
    output_path: Optional[Path] = None,
    phone_model: Optional[str] = None,
    profile_db: Optional[Path] = None,
    fallback_focal_px: Optional[float] = None,
) -> Dict:
    exif_info = extract_exif_info(image_path)
    width = int(exif_info["width"])
    height = int(exif_info["height"])
    profiles = load_phone_profiles(profile_db)
    profile, matched_key = find_matching_profile(profiles, phone_model, exif_info)

    intrinsics = None
    source = None

    if profile is not None:
        intrinsics = intrinsics_from_profile(width, height, profile)
        if intrinsics is not None:
            source = "phone_profile"

    if intrinsics is None:
        intrinsics = intrinsics_from_exif(width, height, exif_info, profile)
        if intrinsics is not None:
            source = "exif"

    if intrinsics is None and fallback_focal_px is not None:
        intrinsics = intrinsics_from_focal_px(width, height, fallback_focal_px)
        source = "fallback_focal_px"

    if intrinsics is None:
        raise RuntimeError(
            "Failed to resolve intrinsics. Provide a phone profile, EXIF with focal metadata, or a fallback focal length."
        )

    intrinsics["source"] = source
    intrinsics["phone_model_input"] = phone_model
    intrinsics["matched_profile_key"] = matched_key
    intrinsics["exif_make"] = exif_info["make"]
    intrinsics["exif_model"] = exif_info["model"]
    intrinsics["exif_lens_model"] = exif_info["lens_model"]
    intrinsics["exif_focal_length_mm"] = exif_info["focal_length_mm"]
    intrinsics["exif_focal_length_35mm_equiv_mm"] = exif_info["focal_length_35mm_equiv_mm"]

    if output_path is not None:
        save_json(output_path, intrinsics)
    return intrinsics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer camera intrinsics from phone metadata, EXIF, or fallback focal length.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phone-model", type=str)
    parser.add_argument("--profile-db", type=Path)
    parser.add_argument("--fallback-focal-px", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intrinsics = resolve_intrinsics_for_image(
        image_path=args.image,
        output_path=args.output,
        phone_model=args.phone_model,
        profile_db=args.profile_db,
        fallback_focal_px=args.fallback_focal_px,
    )
    print(json.dumps(intrinsics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
