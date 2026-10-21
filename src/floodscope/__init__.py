"""
FloodScope AI - Automated Satellite Imagery Analysis for Flood Detection & Change Evaluation.
"""

from floodscope.detection import DetectionConfig, DetectionResult, detect_changes
from floodscope.preprocessing import (
    SpatialAlignmentError,
    align_image_pair,
    compute_geospatial_affine_matrix,
    evaluate_spatial_compatibility,
    extract_geotiff_metadata,
    generate_alignment_diagnostics,
    load_image,
)

__version__ = "0.1.0"

__all__ = [
    "detect_changes",
    "DetectionConfig",
    "DetectionResult",
    "align_image_pair",
    "load_image",
    "extract_geotiff_metadata",
    "evaluate_spatial_compatibility",
    "compute_geospatial_affine_matrix",
    "generate_alignment_diagnostics",
    "SpatialAlignmentError",
]
