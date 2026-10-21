# FloodScope AI — Member 2 Detection Handover Specification
**Author:** Member 2 — Detection (Satellite Preprocessing, Alignment & Change Detection)  
**Recipient:** Member 3 — Application & Integration  
**Date:** October 21, 2026  
**Status:** Baseline Implemented, Verified, and Tested (30/30 Unit & Integration Tests Passed)  

---

## 1. Scope & Implementation Overview

Member 2 has implemented and verified the complete preprocessing, spatial alignment, and OpenCV-based change detection baseline for FloodScope AI.

### Capabilities Delivered:
1. **Multi-Sensor Safe Raster Ingestion:** Robust GeoTIFF loading for RGB optical, 13-band Sentinel-2 multispectral, and 2-band Sentinel-1 float32 SAR.
2. **Sub-Pixel Spatial Alignment:** Mathematical forward affine transformation ($M$) derived directly from GeoTIFF tiepoints and scales in `EPSG:4326`, resampling mismatched captures (e.g. $1743 \times 1743$ to $1300 \times 1300$) without naive resizing or spatial drift.
3. **Valid Footprint Masking:** Rigorous overlap mask generation ensuring invalid, out-of-bounds, or nodata border pixels are strictly excluded from change detections.
4. **Explainable Change Detection:** OpenCV-based luminance/perceptual absolute difference calculation with configurable thresholding, optional morphological cleanup, and visual overlay highlighting.
5. **Standardized Python API:** A clean, typed, documented public interface (`detect_changes`) returning a structured `DetectionResult` dataclass.

---

## 2. Inventory of New and Modified Files

All original raw files in `dataset/raw/` and labels in `dataset/labels/` remain completely unchanged.

### Source Code:
- [`src/floodscope/__init__.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/src/floodscope/__init__.py) — Package public exports (`detect_changes`, `DetectionConfig`, `DetectionResult`, `align_image_pair`, etc.).
- [`src/floodscope/preprocessing.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/src/floodscope/preprocessing.py) — Image loading, GeoTIFF tag extraction, spatial compatibility checking, forward affine warping, and alignment diagnostics.
- [`src/floodscope/detection.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/src/floodscope/detection.py) — Change detection baseline, thresholding, morphological filtering, overlay blending, and result dataclass.

### Execution Scripts:
- [`scripts/run_alignment.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/scripts/run_alignment.py) — Geospatial alignment pipeline runner.
- [`scripts/run_detection.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/scripts/run_detection.py) — Multi-threshold change detection runner on verified real data.
- [`scripts/audit_stage2.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/scripts/audit_stage2.py) — SHA-256 integrity and reproducibility audit script.

### Automated Test Suite:
- [`tests/test_preprocessing.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/tests/test_preprocessing.py) — 15 unit and integration tests for loading, metadata, transforms, diagnostics, and failure handling.
- [`tests/test_detection.py`](file:///c:/Users/priya%20dharshini.S/Downloads/FloodScope_AI/FloodScope_AI/tests/test_detection.py) — 15 unit and integration tests for identical pairs, synthetic changes, invalid-pixel masking, error states, and dataclass schemas.

### Primary Processed Artifacts in `dataset/processed/`:
- `SpaceNet8_Germany_0_16_67_PRE_aligned.png` — Aligned reference capture ($1300 \times 1300 \times 3$, `uint8`).
- `SpaceNet8_Germany_0_16_67_POST_aligned.png` — Aligned post capture ($1300 \times 1300 \times 3$, `uint8`).
- `SpaceNet8_Germany_0_16_67_overlap_mask.png` — Binary valid overlap mask ($1300 \times 1300$, `uint8`).
- `SpaceNet8_Germany_0_16_67_candidate_change_mask.png` — Binary candidate change mask ($255=\text{change}$, $0=\text{no change}$).
- `SpaceNet8_Germany_0_16_67_change_overlay.png` — Red candidate change highlights overlaid on reference imagery.
- `SpaceNet8_Germany_0_16_67_difference_image.png` — Raw intensity difference map ($0\text{--}255$).
- `SpaceNet8_Germany_0_16_67_detection_summary.json` — Machine-readable summary metrics and audit record.

---

## 3. Environment & Dependencies

Tested and verified under:
- **Python:** `3.13.14`
- **OpenCV:** `4.13.0` (`opencv-python 4.13.0.92`)
- **NumPy:** `2.4.2`
- **Tifffile:** `2026.6.1`
- **Test Runner:** Python standard library `unittest` (zero third-party test dependencies required).

No special C-extensions (like GDAL) are required. All GeoTIFF tags and multi-band arrays are read via `tifffile` and `OpenCV`.

---

## 4. Member 3 Public API Contract

### Function Signature:
```python
from floodscope import detect_changes, DetectionConfig, DetectionResult

def detect_changes(
    before_path: Union[str, Path, np.ndarray],
    after_path: Union[str, Path, np.ndarray],
    overlap_mask_path: Optional[Union[str, Path, np.ndarray]] = None,
    config: Optional[DetectionConfig] = None
) -> DetectionResult:
```

### Configuration Dataclass (`DetectionConfig`):
```python
@dataclass
class DetectionConfig:
    threshold: float = 35.0                    # Difference cutoff (0.0 to 255.0)
    method: str = "grayscale_diff"             # 'grayscale_diff' or 'color_diff'
    apply_morphology: bool = True              # Morphological opening noise filter
    morph_kernel_size: int = 3                 # Odd kernel size (e.g. 3)
    overlay_color_bgr: Tuple[int, int, int] = (0, 0, 255)  # Red BGR
    overlay_alpha: float = 0.5                 # Blending transparency (0.0 to 1.0)
```

### Return Dataclass (`DetectionResult`):
| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `str` | `"SUCCESS"` or `"FAILED"`. |
| `change_mask` | `np.ndarray` | 2D `uint8` array ($H, W$), values `255` (candidate change) or `0` (unchanged / invalid). |
| `overlay` | `np.ndarray` | 3D `uint8` array ($H, W, 3$) in BGR format with changes highlighted. |
| `difference_image` | `np.ndarray` | 2D `uint8` array ($H, W$) raw absolute difference map. |
| `summary` | `Dict[str, Any]` | Metrics dict containing `valid_pixels`, `changed_pixels`, `valid_area_change_pct`, `threshold`. |
| `warnings` | `List[str]` | Explicit disclaimers stating changes are not confirmed floodwater. |
| `metadata` | `Dict[str, Any]` | Dimensions, channel layout, config parameters, and OpenCV version. |
| `error` | `Optional[str]` | `None` on success, or a descriptive error string on failure. |

---

## 5. Minimal Usage Example for Member 3

```python
from pathlib import Path
from floodscope import detect_changes, DetectionConfig

# Define input paths
pre_img = Path("dataset/processed/SpaceNet8_Germany_0_16_67_PRE_aligned.png")
post_img = Path("dataset/processed/SpaceNet8_Germany_0_16_67_POST_aligned.png")
mask = Path("dataset/processed/SpaceNet8_Germany_0_16_67_overlap_mask.png")

# Configure detector
config = DetectionConfig(threshold=35.0, apply_morphology=True)

# Run change detection
result = detect_changes(
    before_path=pre_img,
    after_path=post_img,
    overlap_mask_path=mask,
    config=config
)

if result.status == "SUCCESS":
    print(f"Valid Pixels:  {result.summary['valid_pixels']}")
    print(f"Changed Pixels:{result.summary['changed_pixels']}")
    print(f"Change Area %: {result.summary['valid_area_change_pct']:.2f}%")
    
    # Save or pass numpy arrays to frontend UI
    mask_array = result.change_mask    # Shape: (1300, 1300), uint8
    overlay_bgr = result.overlay       # Shape: (1300, 1300, 3), uint8
else:
    print(f"Detection failed: {result.error}")
```

---

## 6. Execution & Verification Commands

### Run Full Test Suite:
```powershell
python -m unittest discover tests -v
```
**Actual Result:** `Ran 30 tests in 0.821s — OK (30 passed, 0 failed, 0 skipped)`.

### Run Real-Data Pipeline:
```powershell
# 1. Geospatial Alignment
python scripts/run_alignment.py

# 2. Change Detection Baseline
python scripts/run_detection.py
```

---

## 7. Critical Scientific & Domain Disclaimers

> [!WARNING]
> **Candidate Changes $\neq$ Confirmed Floodwater**
>
> In the SpaceNet 8 sample tile (`0_16_67`):
> 1. **Atmospheric Clouds:** Low-altitude cloud banks in the northwest of `PRE` create high radiometric deltas against clear ground in `POST`.
> 2. **Phenological Seasonality:** `PRE` captures dormant winter/spring vegetation with bare deciduous trees and frost/pale soil; `POST` captures dense green summer canopy (July 2021).
> 3. **Agricultural Shifts:** Active crop growth and tilled soils cause prominent reflectance differences.
>
> Member 3 must display outputs with clear disclaimers labeling detections as **Candidate Surface Changes** or **Observed Radiometric Deltas**, rather than claiming verified floodwater inundation. Real flood-specific accuracy remains unbenchmarked until independent bi-temporal water ground truth is provided.
