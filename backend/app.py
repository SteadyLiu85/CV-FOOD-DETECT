import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs" / "server"
DEFAULT_CHECKPOINT = ROOT / "models" / "depth_pro.pt"

app = FastAPI(title="Food Volume Estimation API", version="0.1.0")


def _timeout_seconds() -> float:
    return float(os.environ.get("FOOD_VOLUME_TIMEOUT_SECONDS", "10"))


def _checkpoint_path() -> Path:
    return Path(os.environ.get("FOOD_VOLUME_DEPTHPRO_CHECKPOINT", DEFAULT_CHECKPOINT))


async def _save_upload(upload: UploadFile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    path.write_bytes(data)
    return path


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _physical_estimation(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    volume_summary = result.get("volume_summary")
    if not isinstance(volume_summary, dict):
        return None
    physical = volume_summary.get("physical_estimation")
    if not isinstance(physical, dict):
        return None
    return physical


def _estimate_consumed(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_physical = _physical_estimation(before)
    after_physical = _physical_estimation(after)
    if before_physical is None or after_physical is None:
        return {
            "available": False,
            "message": "Before/after volume summaries are incomplete. Check food class and pipeline outputs.",
        }

    metric_keys = ["estimated_volume_ml", "estimated_mass_g", "estimated_kcal"]
    consumed: Dict[str, Any] = {"available": True, "warnings": []}
    raw_delta: Dict[str, Optional[float]] = {}

    for key in metric_keys:
        before_value = before_physical.get(key)
        after_value = after_physical.get(key)
        if before_value is None or after_value is None:
            consumed[key] = None
            raw_delta[key] = None
            consumed["warnings"].append(f"Missing {key}; cannot compute this consumed metric.")
            continue

        delta = float(before_value) - float(after_value)
        raw_delta[key] = delta
        consumed[key] = max(delta, 0.0)
        if delta < 0:
            consumed["warnings"].append(
                f"After-meal {key} is larger than before-meal {key}; clamped consumed value to 0."
            )

    before_kcal = before_physical.get("estimated_kcal")
    consumed_kcal = consumed.get("estimated_kcal")
    if before_kcal and consumed_kcal is not None and float(before_kcal) > 0:
        consumed["intake_ratio"] = max(min(float(consumed_kcal) / float(before_kcal), 1.0), 0.0)
    else:
        consumed["intake_ratio"] = None

    consumed["raw_delta"] = raw_delta
    return consumed


def _run_pipeline(
    image_path: Path,
    output_dir: Path,
    food: Optional[str],
    phone_model: Optional[str],
    prompt: Optional[str],
    mask_path: Optional[Path],
    intrinsics_path: Optional[Path],
) -> Dict[str, Any]:
    checkpoint = _checkpoint_path()
    if not checkpoint.exists():
        raise HTTPException(status_code=500, detail=f"DepthPro checkpoint not found: {checkpoint}")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_auto_food_volume_demo.py"),
        "--checkpoint",
        str(checkpoint),
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--log-file",
        str(output_dir / "pipeline.log.md"),
    ]
    if food:
        cmd.extend(["--food", food])
    if phone_model:
        cmd.extend(["--phone-model", phone_model])
    if prompt:
        cmd.extend(["--prompt", prompt])
    if mask_path:
        cmd.extend(["--mask", str(mask_path)])
    if intrinsics_path:
        cmd.extend(["--intrinsics", str(intrinsics_path)])

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "message": "Food volume inference timed out.",
                "timeout_seconds": _timeout_seconds(),
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        ) from exc

    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Food volume pipeline failed.",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "output_dir": str(output_dir),
            },
        )

    pipeline_summary = _load_json(output_dir / "pipeline_summary.json")
    volume_summary_path = None
    volume_summary = None
    if pipeline_summary and pipeline_summary.get("volume_summary"):
        volume_summary_path = Path(pipeline_summary["volume_summary"])
        volume_summary = _load_json(volume_summary_path)

    return {
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "pipeline_summary": pipeline_summary,
        "volume_summary": volume_summary,
        "stdout": completed.stdout,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    checkpoint = _checkpoint_path()
    return {
        "status": "ok",
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint": str(checkpoint),
        "timeout_seconds": _timeout_seconds(),
    }


@app.post("/api/v1/food-volume")
async def estimate_food_volume(
    image: UploadFile = File(...),
    food: Optional[str] = Form(None),
    phone_model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
    intrinsics: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    request_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = RUNS_ROOT / request_id
    request_dir = job_dir / "00_request"

    image_suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    image_path = await _save_upload(image, request_dir / f"input{image_suffix}")

    mask_path = None
    if mask is not None:
        mask_suffix = Path(mask.filename or "mask.png").suffix or ".png"
        mask_path = await _save_upload(mask, request_dir / f"mask{mask_suffix}")

    intrinsics_path = None
    if intrinsics is not None:
        intrinsics_path = await _save_upload(intrinsics, request_dir / "intrinsics.json")

    result = _run_pipeline(
        image_path=image_path,
        output_dir=job_dir,
        food=food,
        phone_model=phone_model,
        prompt=prompt,
        mask_path=mask_path,
        intrinsics_path=intrinsics_path,
    )
    return {"request_id": request_id, **result}


@app.post("/api/v1/food-consumption")
async def estimate_food_consumption(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
    food: Optional[str] = Form(None),
    phone_model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    before_mask: Optional[UploadFile] = File(None),
    after_mask: Optional[UploadFile] = File(None),
    before_intrinsics: Optional[UploadFile] = File(None),
    after_intrinsics: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    request_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = RUNS_ROOT / request_id
    request_dir = job_dir / "00_request"

    before_suffix = Path(before_image.filename or "before.jpg").suffix or ".jpg"
    after_suffix = Path(after_image.filename or "after.jpg").suffix or ".jpg"
    before_image_path = await _save_upload(before_image, request_dir / f"before{before_suffix}")
    after_image_path = await _save_upload(after_image, request_dir / f"after{after_suffix}")

    before_mask_path = None
    if before_mask is not None:
        before_mask_suffix = Path(before_mask.filename or "before_mask.png").suffix or ".png"
        before_mask_path = await _save_upload(before_mask, request_dir / f"before_mask{before_mask_suffix}")

    after_mask_path = None
    if after_mask is not None:
        after_mask_suffix = Path(after_mask.filename or "after_mask.png").suffix or ".png"
        after_mask_path = await _save_upload(after_mask, request_dir / f"after_mask{after_mask_suffix}")

    before_intrinsics_path = None
    if before_intrinsics is not None:
        before_intrinsics_path = await _save_upload(before_intrinsics, request_dir / "before_intrinsics.json")

    after_intrinsics_path = None
    if after_intrinsics is not None:
        after_intrinsics_path = await _save_upload(after_intrinsics, request_dir / "after_intrinsics.json")

    started = time.perf_counter()
    before_result = _run_pipeline(
        image_path=before_image_path,
        output_dir=job_dir / "before",
        food=food,
        phone_model=phone_model,
        prompt=prompt,
        mask_path=before_mask_path,
        intrinsics_path=before_intrinsics_path,
    )
    after_result = _run_pipeline(
        image_path=after_image_path,
        output_dir=job_dir / "after",
        food=food,
        phone_model=phone_model,
        prompt=prompt,
        mask_path=after_mask_path,
        intrinsics_path=after_intrinsics_path,
    )
    consumed = _estimate_consumed(before_result, after_result)
    elapsed = time.perf_counter() - started

    response = {
        "request_id": request_id,
        "elapsed_seconds": elapsed,
        "output_dir": str(job_dir),
        "before": before_result,
        "after": after_result,
        "consumed": consumed,
    }
    (job_dir / "consumption_summary.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return response
