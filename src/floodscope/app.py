"""
FloodScope AI - Application Backend & API Service
Member 3: Application & Integration

FastAPI backend service connecting Member 2's candidate change detection
baseline with the web dashboard. Provides:
1. Health and diagnostics endpoints.
2. Sample dataset metadata and precomputed baseline results.
3. Synchronous detection execution on verified sample tiles with custom parameters.
4. Upload-based detection on user-provided before/after satellite image pairs.
5. Secure static asset and artifact serving with path traversal protections.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure src/ is importable
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

from floodscope.detection import (
    DetectionConfig,
    DetectionResult,
    detect_changes,
    save_detection_artifacts,
)

PROCESSED_DIR = REPO_ROOT / "dataset" / "processed"
RUNS_DIR = PROCESSED_DIR / "runs"
STATIC_DIR = APP_DIR / "static"

# Ensure runs directory exists
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="FloodScope AI API",
    description="Automated Satellite Imagery Analysis for Candidate Surface Change Detection",
    version="0.1.0",
)

# Enable CORS for local testing and developer friendliness
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit per image


# -------------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------------

class DetectSampleRequest(BaseModel):
    sample_id: str = Field(default="SpaceNet8_Germany_0_16_67", description="ID of verified sample dataset tile")
    threshold: float = Field(default=35.0, ge=0.0, le=255.0, description="Intensity difference threshold (0-255)")
    method: str = Field(default="grayscale_diff", description="'grayscale_diff' or 'color_diff'")
    apply_morphology: bool = Field(default=True, description="Apply morphological opening noise filter")
    morph_kernel_size: int = Field(default=3, description="Odd integer kernel size for morphology")


# -------------------------------------------------------------------------
# Security & Path Helpers
# -------------------------------------------------------------------------

SAFE_RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+\.(png|jpg|jpeg|json)$", re.IGNORECASE)


def validate_safe_path(base_dir: Path, subpath: Path) -> Path:
    """Ensures subpath resolves strictly within base_dir to block path traversal."""
    resolved_base = base_dir.resolve()
    resolved_target = subpath.resolve()
    if not str(resolved_target).startswith(str(resolved_base)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Path traversal attempt detected."
        )
    return resolved_target


def safe_decode_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Safely decodes an image from byte buffer, rejecting corrupt or unsupported data."""
    if not raw_bytes or len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image file '{filename}' is empty (0 bytes)."
        )
    if len(raw_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file '{filename}' exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB."
        )

    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode '{filename}'. The file is corrupted or not a supported image format."
        )
    return img


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Health check and dependency verification endpoint."""
    pre_aligned = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_PRE_aligned.png"
    post_aligned = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_POST_aligned.png"
    sample_ready = pre_aligned.is_file() and post_aligned.is_file()

    return {
        "status": "healthy",
        "service": "FloodScope AI Candidate Change Detection Service",
        "version": "0.1.0",
        "opencv_version": cv2.__version__,
        "python_version": sys.version.split()[0],
        "sample_data_available": sample_ready,
    }


@app.get("/api/sample-datasets")
def get_sample_datasets() -> Dict[str, Any]:
    """Returns available sample datasets, precomputed baseline artifacts, and diagnostics."""
    summary_path = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_detection_summary.json"
    baseline_summary = None
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                baseline_summary = json.load(f)
        except Exception:
            baseline_summary = None

    return {
        "samples": [
            {
                "id": "SpaceNet8_Germany_0_16_67",
                "name": "SpaceNet 8 Germany (Tile 0_16_67)",
                "description": "Bi-temporal optical satellite pair aligned to 1300x1300 grid (EPSG:4326).",
                "default_threshold": 35.0,
                "artifacts": {
                    "pre_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_PRE_aligned.png",
                    "post_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_POST_aligned.png",
                    "overlap_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_overlap_mask.png",
                    "candidate_change_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_candidate_change_mask.png",
                    "change_overlay_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_change_overlay.png",
                    "difference_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_difference_image.png",
                    "alignment_side_by_side_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_alignment_side_by_side.png",
                    "alignment_checkerboard_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_alignment_checkerboard.png",
                    "alignment_overlay_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_alignment_overlay.png",
                },
                "baseline_summary": baseline_summary,
            }
        ]
    }


@app.post("/api/detect/sample")
def detect_sample_changes(req: DetectSampleRequest) -> Dict[str, Any]:
    """
    Executes Member 2's detect_changes() on the verified SpaceNet 8 sample dataset.
    Preserves exact metrics, metadata, and scientific disclaimers.
    """
    if req.sample_id != "SpaceNet8_Germany_0_16_67":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample dataset '{req.sample_id}' not found."
        )

    pre_path = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_PRE_aligned.png"
    post_path = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_POST_aligned.png"
    mask_path = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_overlap_mask.png"

    if not pre_path.is_file() or not post_path.is_file() or not mask_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Aligned sample files are missing from dataset/processed/."
        )

    # Validate DetectionConfig parameters
    try:
        cfg = DetectionConfig(
            threshold=req.threshold,
            method=req.method,
            apply_morphology=req.apply_morphology,
            morph_kernel_size=req.morph_kernel_size,
        )
        cfg.validate()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid detection configuration: {e}"
        )

    # Check if default parameters match the pre-computed baseline
    is_default_baseline = (
        req.threshold == 35.0
        and req.method == "grayscale_diff"
        and req.apply_morphology is True
        and req.morph_kernel_size == 3
    )

    if is_default_baseline:
        summary_path = PROCESSED_DIR / "SpaceNet8_Germany_0_16_67_detection_summary.json"
        if summary_path.is_file():
            with open(summary_path, "r", encoding="utf-8") as f:
                cached_res = json.load(f)
            return {
                "status": cached_res.get("status", "SUCCESS"),
                "run_id": "baseline",
                "summary": cached_res.get("summary", {}),
                "warnings": cached_res.get("warnings", []),
                "metadata": cached_res.get("metadata", {}),
                "artifacts": {
                    "pre_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_PRE_aligned.png",
                    "post_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_POST_aligned.png",
                    "overlap_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_overlap_mask.png",
                    "change_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_candidate_change_mask.png",
                    "overlay_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_change_overlay.png",
                    "difference_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_difference_image.png",
                },
                "error": None,
            }

    # Run detection dynamically
    result: DetectionResult = detect_changes(
        before_path=pre_path,
        after_path=post_path,
        overlap_mask_path=mask_path,
        config=cfg,
    )

    if result.status != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Detection algorithm failed."
        )

    # Save run-specific artifacts
    run_id = f"sample_{uuid.uuid4().hex[:10]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = save_detection_artifacts(result, run_dir, prefix="result")

    return {
        "status": result.status,
        "run_id": run_id,
        "summary": result.summary,
        "warnings": result.warnings,
        "metadata": result.metadata,
        "artifacts": {
            "pre_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_PRE_aligned.png",
            "post_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_POST_aligned.png",
            "overlap_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_overlap_mask.png",
            "change_mask_url": f"/api/outputs/{run_id}/result_candidate_change_mask.png",
            "overlay_url": f"/api/outputs/{run_id}/result_change_overlay.png",
            "difference_image_url": f"/api/outputs/{run_id}/result_difference_image.png",
        },
        "error": None,
    }


@app.post("/api/detect/upload")
async def detect_uploaded_changes(
    before_image: UploadFile = File(..., description="Pre-event aligned image"),
    after_image: UploadFile = File(..., description="Post-event aligned image"),
    overlap_mask: Optional[UploadFile] = File(None, description="Optional binary overlap mask"),
    threshold: float = Form(35.0),
    method: str = Form("grayscale_diff"),
    apply_morphology: bool = Form(True),
    morph_kernel_size: int = Form(3),
) -> Dict[str, Any]:
    """
    Accepts uploaded before/after satellite images, validates dimensions and formats,
    executes Member 2's detect_changes(), and returns candidate surface changes.
    """
    # 1. Validate configuration
    try:
        cfg = DetectionConfig(
            threshold=threshold,
            method=method,
            apply_morphology=apply_morphology,
            morph_kernel_size=morph_kernel_size,
        )
        cfg.validate()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid detection configuration: {e}"
        )

    # 2. Read and decode images safely
    raw_before = await before_image.read()
    raw_after = await after_image.read()

    img_pre = safe_decode_image(raw_before, before_image.filename or "before_image")
    img_post = safe_decode_image(raw_after, after_image.filename or "after_image")

    # Validate dimension compatibility
    if img_pre.shape[:2] != img_post.shape[:2]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image dimensions mismatch: PRE={img_pre.shape[:2]} vs POST={img_post.shape[:2]}. "
                f"Satellite images must be pre-aligned onto identical pixel grids."
            )
        )

    mask_arr: Optional[np.ndarray] = None
    if overlap_mask is not None and overlap_mask.filename:
        raw_mask = await overlap_mask.read()
        if len(raw_mask) > 0:
            mask_arr = safe_decode_image(raw_mask, overlap_mask.filename)
            if mask_arr.shape[:2] != img_pre.shape[:2]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Overlap mask dimensions {mask_arr.shape[:2]} do not match image dimensions {img_pre.shape[:2]}."
                )

    # 3. Execute detect_changes
    result: DetectionResult = detect_changes(
        before_path=img_pre,
        after_path=img_post,
        overlap_mask_path=mask_arr,
        config=cfg,
    )

    if result.status != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.error or "Change detection failed on uploaded imagery."
        )

    # 4. Save results to unique run directory
    run_id = f"upload_{uuid.uuid4().hex[:10]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs for visual inspection
    cv2.imwrite(str(run_dir / "uploaded_PRE.png"), img_pre)
    cv2.imwrite(str(run_dir / "uploaded_POST.png"), img_post)
    if mask_arr is not None:
        cv2.imwrite(str(run_dir / "uploaded_mask.png"), mask_arr)

    # Save output artifacts
    save_detection_artifacts(result, run_dir, prefix="result")

    return {
        "status": result.status,
        "run_id": run_id,
        "summary": result.summary,
        "warnings": result.warnings,
        "metadata": result.metadata,
        "artifacts": {
            "pre_image_url": f"/api/outputs/{run_id}/uploaded_PRE.png",
            "post_image_url": f"/api/outputs/{run_id}/uploaded_POST.png",
            "overlap_mask_url": f"/api/outputs/{run_id}/uploaded_mask.png" if mask_arr is not None else None,
            "change_mask_url": f"/api/outputs/{run_id}/result_candidate_change_mask.png",
            "overlay_url": f"/api/outputs/{run_id}/result_change_overlay.png",
            "difference_image_url": f"/api/outputs/{run_id}/result_difference_image.png",
        },
        "error": None,
    }


@app.get("/api/outputs/{run_id}/{filename}")
def get_output_artifact(run_id: str, filename: str) -> FileResponse:
    """
    Safely serves output artifacts (images and JSON summaries) with path traversal checks.
    """
    if not SAFE_RUN_ID_PATTERN.match(run_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid run ID format.")
    if not SAFE_FILENAME_PATTERN.match(filename):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact filename.")

    if run_id == "baseline":
        target_path = PROCESSED_DIR / filename
        validate_safe_path(PROCESSED_DIR, target_path)
    else:
        target_path = RUNS_DIR / run_id / filename
        validate_safe_path(RUNS_DIR, target_path)

    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{filename}' not found for run '{run_id}'."
        )

    # Determine media type
    suffix = target_path.suffix.lower()
    media_type = "image/png"
    if suffix in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif suffix == ".json":
        media_type = "application/json"

    return FileResponse(path=str(target_path), media_type=media_type)


# -------------------------------------------------------------------------
# Static Dashboard & UI Routes
# -------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page() -> HTMLResponse:
    """Serves the main FloodScope AI frontend dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return HTMLResponse(
            "<h3>FloodScope AI API Service Running</h3><p>Static dashboard files not found in src/floodscope/static.</p>",
            status_code=200,
        )
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    """Factory helper for ASGI server runners."""
    return app
