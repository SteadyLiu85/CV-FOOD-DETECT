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
