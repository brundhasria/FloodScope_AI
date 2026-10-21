"""
FloodScope AI - Change Detection Baseline Module
Member 2: Detection (Image Comparison, Candidate Change Extraction, and Diagnostics)

This module provides:
1. Explainable, reproducible OpenCV-based candidate change detection between aligned PRE and POST scenes.
2. Rigorous overlap-mask integration: invalid/border pixels are strictly excluded from detections and statistics.
3. Configurable difference algorithms (grayscale luminance delta or multi-band Euclidean color distance).
4. Morphological noise suppression and configurable thresholding.
5. Standardized, stable DetectionResult contract for Member 3 application integration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class DetectionInputError(ValueError):
    """Raised when detection input images or masks are missing, corrupted, or dimensionally incompatible."""
    pass


@dataclass
class DetectionConfig:
    """
    Configuration parameters for the change detection baseline.

    Attributes:
        threshold: Intensity difference threshold (0.0 to 255.0). Default: 35.0.
        method: Difference computation method ('grayscale_diff' or 'color_diff'). Default: 'grayscale_diff'.
        apply_morphology: Whether to apply modest morphological noise filtering. Default: True.
        morph_kernel_size: Odd integer kernel size for morphological opening. Default: 3.
        overlay_color_bgr: Tuple of (B, G, R) for highlighting candidate changes in overlay. Default: (0, 0, 255) [Red].
        overlay_alpha: Blending weight of candidate change highlight (0.0 to 1.0). Default: 0.5.
    """
    threshold: float = 35.0
    method: str = "grayscale_diff"
    apply_morphology: bool = True
    morph_kernel_size: int = 3
    overlay_color_bgr: Tuple[int, int, int] = (0, 0, 255)
    overlay_alpha: float = 0.5

    def validate(self) -> None:
        if self.threshold < 0.0 or self.threshold > 255.0:
            raise ValueError(f"Threshold must be between 0.0 and 255.0, got {self.threshold}")
        if self.method not in ["grayscale_diff", "color_diff"]:
            raise ValueError(f"Unknown method '{self.method}'. Must be 'grayscale_diff' or 'color_diff'")
        if self.morph_kernel_size < 1 or self.morph_kernel_size % 2 == 0:
            raise ValueError(f"morph_kernel_size must be a positive odd integer, got {self.morph_kernel_size}")
        if not (0.0 <= self.overlay_alpha <= 1.0):
            raise ValueError(f"overlay_alpha must be between 0.0 and 1.0, got {self.overlay_alpha}")


@dataclass
class DetectionResult:
    """
    Standardized return object for change detection. Member 3 integrates against this contract.

    Attributes:
        status: "SUCCESS" or "FAILED".
        change_mask: 2D uint8 numpy array (H, W) where 255 indicates candidate change, 0 unchanged/invalid.
        overlay: 3D uint8 numpy array (H, W, 3) with candidate changes highlighted on reference image.
        difference_image: 2D uint8 numpy array (H, W) raw difference map before thresholding.
        summary: Dictionary of quantitative metrics (pixel counts, percentages, threshold).
        warnings: List of explicit limitations and interpretation disclaimers.
        metadata: Detailed record of execution parameters, dimensions, and library versions.
        error: Descriptive error message if status == "FAILED", otherwise None.
    """
    status: str
    change_mask: np.ndarray
    overlay: np.ndarray
    difference_image: np.ndarray
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary summary (excluding heavy numpy arrays)."""
        return {
            "status": self.status,
            "summary": self.summary,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "error": self.error,
        }


def _load_array(input_data: Union[str, Path, np.ndarray], name: str) -> np.ndarray:
    """Helper to safely read an image from path or validate an existing numpy array."""
    if isinstance(input_data, (str, Path)):
        p = Path(input_data)
        if not p.is_file():
            raise DetectionInputError(f"{name} file does not exist: {p.resolve()}")
        if p.stat().st_size == 0:
            raise DetectionInputError(f"{name} file is empty (0 bytes): {p.resolve()}")
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise DetectionInputError(f"Failed to decode image from {name}: {p.resolve()}")
        return arr
    elif isinstance(input_data, np.ndarray):
        if input_data.size == 0:
            raise DetectionInputError(f"{name} array is empty (0 elements)")
        return input_data.copy()
    else:
        raise DetectionInputError(f"{name} must be a file path (str/Path) or numpy array, got {type(input_data)}")


def detect_changes(
    before_path: Union[str, Path, np.ndarray],
    after_path: Union[str, Path, np.ndarray],
    overlap_mask_path: Optional[Union[str, Path, np.ndarray]] = None,
    config: Optional[DetectionConfig] = None
) -> DetectionResult:
    """
    Executes explainable OpenCV change detection between aligned PRE and POST satellite imagery.

    Parameters:
        before_path: Aligned pre-event image (file path or numpy array).
        after_path: Aligned post-event image (file path or numpy array).
        overlap_mask_path: Optional binary validity mask (255=valid, 0=invalid/nodata).
        config: Optional DetectionConfig instance with algorithm settings.

    Returns:
        DetectionResult dataclass containing change mask, visual overlay, difference map, and audit metrics.
    """
    cfg = config or DetectionConfig()
    try:
        cfg.validate()
    except Exception as e:
        empty_mask = np.zeros((0, 0), dtype=np.uint8)
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        return DetectionResult(
            status="FAILED",
            change_mask=empty_mask,
            overlay=empty_img,
            difference_image=empty_mask,
            error=f"Invalid DetectionConfig: {e}"
        )

    # 1. Load and validate inputs
    try:
        img_pre = _load_array(before_path, "before_image")
        img_post = _load_array(after_path, "after_image")
    except DetectionInputError as err:
        empty_mask = np.zeros((0, 0), dtype=np.uint8)
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        return DetectionResult(
            status="FAILED",
            change_mask=empty_mask,
            overlay=empty_img,
            difference_image=empty_mask,
            error=str(err)
        )

    # Dimension and channel compatibility checks
    if img_pre.shape[:2] != img_post.shape[:2]:
        err_msg = (
            f"Image dimensions mismatch: PRE={img_pre.shape[:2]} vs POST={img_post.shape[:2]}. "
            f"Input images must be pre-aligned onto a common spatial grid."
        )
        empty_mask = np.zeros((0, 0), dtype=np.uint8)
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        return DetectionResult(
            status="FAILED",
            change_mask=empty_mask,
            overlay=empty_img,
            difference_image=empty_mask,
            error=err_msg
        )

    h, w = img_pre.shape[:2]

    # Validate overlap mask
    if overlap_mask_path is not None:
        try:
            mask = _load_array(overlap_mask_path, "overlap_mask")
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            if mask.shape != (h, w):
                err_msg = f"Overlap mask dimensions {mask.shape} do not match image dimensions {(h, w)}."
                empty_mask = np.zeros((0, 0), dtype=np.uint8)
                empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
                return DetectionResult(
                    status="FAILED",
                    change_mask=empty_mask,
                    overlay=empty_img,
                    difference_image=empty_mask,
                    error=err_msg
                )
            # Ensure boolean validity mask (True where mask == 255 or > 0)
            valid_mask = (mask > 0)
        except DetectionInputError as err:
            empty_mask = np.zeros((0, 0), dtype=np.uint8)
            empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
            return DetectionResult(
                status="FAILED",
                change_mask=empty_mask,
                overlay=empty_img,
                difference_image=empty_mask,
                error=str(err)
            )
    else:
        valid_mask = np.ones((h, w), dtype=bool)

    # 2. Compute difference image based on configured method
    if cfg.method == "grayscale_diff":
        # Convert BGR/RGB to grayscale luminance
        if img_pre.ndim == 3 and img_pre.shape[2] >= 3:
            gray_pre = cv2.cvtColor(img_pre[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif img_pre.ndim == 2:
            gray_pre = img_pre
        else:
            gray_pre = img_pre[:, :, 0]

        if img_post.ndim == 3 and img_post.shape[2] >= 3:
            gray_post = cv2.cvtColor(img_post[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif img_post.ndim == 2:
            gray_post = img_post
        else:
            gray_post = img_post[:, :, 0]

        diff = cv2.absdiff(gray_pre, gray_post)

    elif cfg.method == "color_diff":
        # Per-band Euclidean distance: sqrt(sum((C_pre - C_post)^2))
        p1 = img_pre[:, :, :3].astype(np.float32) if img_pre.ndim == 3 else img_pre.astype(np.float32)
        p2 = img_post[:, :, :3].astype(np.float32) if img_post.ndim == 3 else img_post.astype(np.float32)
        if p1.ndim == 3:
            diff_f = np.sqrt(np.sum((p1 - p2) ** 2, axis=2)) / np.sqrt(3.0)
        else:
            diff_f = np.abs(p1 - p2)
        diff = np.clip(diff_f, 0, 255).astype(np.uint8)

    # 3. Apply threshold to produce candidate-change mask
    _, raw_thresh = cv2.threshold(diff, int(cfg.threshold), 255, cv2.THRESH_BINARY)

    # 4. Optional morphological cleanup (opening removes isolated speckles)
    if cfg.apply_morphology and cfg.morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_kernel_size, cfg.morph_kernel_size))
        cleaned = cv2.morphologyEx(raw_thresh, cv2.MORPH_OPEN, kernel)
    else:
        cleaned = raw_thresh

    # 5. Mask out all invalid pixels
    # Invalid pixels must NEVER be positive detections
    final_change_mask = np.zeros((h, w), dtype=np.uint8)
    final_change_mask[valid_mask & (cleaned == 255)] = 255

    # 6. Generate visual overlay
    # Format reference image for visualization (uint8 BGR)
    if img_post.dtype != np.uint8:
        ref_norm = np.clip(img_post, 0, 255).astype(np.uint8)
    else:
        ref_norm = img_post.copy()

    if ref_norm.ndim == 2:
        ref_vis = cv2.cvtColor(ref_norm, cv2.COLOR_GRAY2BGR)
    elif ref_norm.shape[2] == 1:
        ref_vis = cv2.cvtColor(ref_norm[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        ref_vis = ref_norm[:, :, :3].copy()

    # Create overlay: colorize candidate changes
    overlay = ref_vis.copy()
    change_pixels = (final_change_mask == 255)
    if np.any(change_pixels):
        color_layer = np.full_like(ref_vis, cfg.overlay_color_bgr, dtype=np.uint8)
        blended = cv2.addWeighted(ref_vis, 1.0 - cfg.overlay_alpha, color_layer, cfg.overlay_alpha, 0.0)
        overlay[change_pixels] = blended[change_pixels]

    # 7. Compute quantitative summary statistics strictly on valid pixels
    total_pixels = int(h * w)
    valid_pixels = int(np.count_nonzero(valid_mask))
    invalid_pixels = total_pixels - valid_pixels
    changed_pixels = int(np.count_nonzero(final_change_mask == 255))

    valid_area_change_pct = (changed_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0

    summary = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "changed_pixels": changed_pixels,
        "valid_area_change_pct": round(float(valid_area_change_pct), 4),
        "threshold": float(cfg.threshold),
        "method": cfg.method,
        "morphology_applied": cfg.apply_morphology,
        "morph_kernel_size": cfg.morph_kernel_size,
    }

    warnings = [
        "DISCLAIMER: Detections represent generic radiometric/surface candidate changes, NOT confirmed floodwater.",
        "Candidate changes may arise from atmospheric cloud cover, shadows, seasonal canopy shifts, agriculture, or sensor geometry.",
        "Ground-truth flood accuracy cannot be verified without independent, validated bi-temporal flood reference labels."
    ]

    metadata = {
        "dimensions": [h, w],
        "channels": int(img_pre.shape[2]) if img_pre.ndim == 3 else 1,
        "opencv_version": cv2.__version__,
        "input_pre_type": str(type(before_path)),
        "input_post_type": str(type(after_path)),
        "config": asdict(cfg),
    }

    return DetectionResult(
        status="SUCCESS",
        change_mask=final_change_mask,
        overlay=overlay,
        difference_image=diff,
        summary=summary,
        warnings=warnings,
        metadata=metadata,
        error=None,
    )


def save_detection_artifacts(
    result: DetectionResult,
    output_dir: Union[str, Path],
    prefix: str = "SpaceNet8_Germany_0_16_67"
) -> Dict[str, Path]:
    """
    Saves detection artifacts (mask, overlay, difference map, and JSON summary) to disk.

    Parameters:
        result: DetectionResult instance from detect_changes().
        output_dir: Target output directory.
        prefix: Naming prefix for generated files.

    Returns:
        Dict mapping artifact keys to their saved Path objects.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    p_mask = out_path / f"{prefix}_candidate_change_mask.png"
    cv2.imwrite(str(p_mask), result.change_mask)
    paths["change_mask"] = p_mask

    p_overlay = out_path / f"{prefix}_change_overlay.png"
    cv2.imwrite(str(p_overlay), result.overlay)
    paths["overlay"] = p_overlay

    p_diff = out_path / f"{prefix}_difference_image.png"
    cv2.imwrite(str(p_diff), result.difference_image)
    paths["difference_image"] = p_diff

    p_summary = out_path / f"{prefix}_detection_summary.json"
    with open(p_summary, "w", encoding="utf-8") as f:
        json.dump(result.to_summary_dict(), f, indent=2)
    paths["summary"] = p_summary

    return paths
