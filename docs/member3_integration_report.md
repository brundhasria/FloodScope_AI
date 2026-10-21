# FloodScope AI — Member 3 Integration & Handover Report

**Author:** Member 3 — Application & Integration  
**Date:** October 21, 2026  
**Status:** Complete, Verified, and Tested (48/48 Tests Passed)

---

## 1. Project Architecture Discovered

Upon initial audit of the workspace:
- **Codebase Root:** `C:\Users\BRUNDHA\Downloads\FloodScope_AI-20261021T145943Z-1-001\FloodScope_AI`
- **Member 1 Assets:** Verified raw bi-temporal and multi-sensor datasets in `dataset/raw/` (SpaceNet 8 optical pair and Sen1Floods11 Bolivia chips) and labels in `dataset/labels/`.
- **Member 2 Assets:**
  - Robust preprocessing, metadata extraction, and sub-pixel spatial alignment in `src/floodscope/preprocessing.py`.
  - Explainable OpenCV-based candidate change detection baseline in `src/floodscope/detection.py`.
  - Processed artifacts in `dataset/processed/` including aligned rasters, difference images, binary masks, and red overlays.
  - 30 unit and integration tests in `tests/test_preprocessing.py` and `tests/test_detection.py`.
- **Identified Gap:** The repository contained no frontend or backend application entry point. Member 3 was tasked with building the application interface, user workflows, and integrating Member 2's verified detection baseline.

---

## 2. Integration Approach Selected & Rationale

We selected a **FastAPI backend with an embedded single-page vanilla web dashboard**:
1. **Zero New External Dependencies:** `fastapi` (0.111.0) and `uvicorn` (0.30.1) were already installed in the user's Python environment. No Node.js build pipelines, bundling steps, or heavy web frameworks were required.
2. **Direct Python Interoperability:** FastAPI directly imports and executes Member 2's public contract (`detect_changes(before_path, after_path, overlap_mask_path, config)`), avoiding fragile cross-process IPC or subprocessing.
3. **Preset & Upload Workflows:** Supports both instant inspection of verified sample tiles (SpaceNet 8 Germany `0_16_67`) and user uploads of custom satellite image pairs.
4. **Security by Design:** Enforces strict image size limits (25MB), file validation against corrupted bytes, dimension compatibility checks, safe filename regex validation, and directory containment checks blocking path traversal attacks.

---

## 3. Files Added or Modified

All original raw files under `dataset/raw/` and labels under `dataset/labels/` remain completely unchanged and pristine. Member 2's baseline modules were preserved.

### Files Added:
- [`src/floodscope/app.py`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/src/floodscope/app.py) — FastAPI backend service implementing health, sample metadata, detection, upload, and secure artifact delivery endpoints.
- [`src/floodscope/static/index.html`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/src/floodscope/static/index.html) — Interactive single-page web dashboard with multi-view tabs, accessible controls, and prominent scientific disclaimers.
- [`src/floodscope/static/style.css`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/src/floodscope/static/style.css) — Responsive dark-themed styling, metric display cards, and layout grids.
- [`src/floodscope/static/app.js`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/src/floodscope/static/app.js) — Client-side controller handling parameter sliders, API communication, multi-view image rendering, and live metric updates.
- [`scripts/run_app.py`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/scripts/run_app.py) — CLI launcher for the Uvicorn web server.
- [`tests/test_app.py`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/tests/test_app.py) — 18 unit and integration tests for all API routes, error conditions, and security guards.
- [`docs/member3_integration_report.md`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/docs/member3_integration_report.md) — This formal integration report.

### Files Modified:
- [`README.md`](file:///C:/Users/BRUNDHA/Downloads/FloodScope_AI-20261021T145943Z-1-001/FloodScope_AI/README.md) — Updated with application execution instructions, API reference, test commands, and scientific guidelines.

---

## 4. API Request & Response Contracts

### 4.1 Sample Detection Endpoint: `POST /api/detect/sample`

**Request:**
```json
POST /api/detect/sample HTTP/1.1
Content-Type: application/json

{
  "sample_id": "SpaceNet8_Germany_0_16_67",
  "threshold": 35.0,
  "method": "grayscale_diff",
  "apply_morphology": true,
  "morph_kernel_size": 3
}
```

**Response (HTTP 200 OK):**
```json
{
  "status": "SUCCESS",
  "run_id": "baseline",
  "summary": {
    "total_pixels": 1690000,
    "valid_pixels": 1690000,
    "invalid_pixels": 0,
    "changed_pixels": 477042,
    "valid_area_change_pct": 28.2273,
    "threshold": 35.0,
    "method": "grayscale_diff",
    "morphology_applied": true,
    "morph_kernel_size": 3
  },
  "warnings": [
    "DISCLAIMER: Detections represent generic radiometric/surface candidate changes, NOT confirmed floodwater.",
    "Candidate changes may arise from atmospheric cloud cover, shadows, seasonal canopy shifts, agriculture, or sensor geometry.",
    "Ground-truth flood accuracy cannot be verified without independent, validated bi-temporal flood reference labels."
  ],
  "metadata": {
    "dimensions": [1300, 1300],
    "channels": 3,
    "opencv_version": "4.13.0"
  },
  "artifacts": {
    "pre_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_PRE_aligned.png",
    "post_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_POST_aligned.png",
    "overlap_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_overlap_mask.png",
    "change_mask_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_candidate_change_mask.png",
    "overlay_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_change_overlay.png",
    "difference_image_url": "/api/outputs/baseline/SpaceNet8_Germany_0_16_67_difference_image.png"
  },
  "error": null
}
```

### 4.2 Image Upload Endpoint: `POST /api/detect/upload`

**Request:** `multipart/form-data` with fields `before_image`, `after_image`, optional `overlap_mask`, and parameters `threshold=35.0`, `method=grayscale_diff`, `apply_morphology=true`.

**Error Behavior (HTTP 400 Bad Request):**
```json
{
  "detail": "Image dimensions mismatch: PRE=(100, 100) vs POST=(120, 120). Satellite images must be pre-aligned onto identical pixel grids."
}
```

---

## 5. Dependency & Environment Verification

- **Python Version:** `Python 3.14.3` (default via `py` launcher; also verified compatible with `3.13.14`).
- **OpenCV Version:** `5.0.0` (`opencv-python 5.0.0.93`).
- **FastAPI Version:** `0.111.0`.
- **Uvicorn Version:** `0.30.1`.
- **NumPy Version:** `2.5.0`.
- **Tifffile Version:** `2026.6.1`.
- **Third-Party Test Tools:** None required; standard library `unittest` was used exclusively.

---

## 6. Tests Executed & Verified Results

Full test suite execution command:
```powershell
python -m unittest discover tests -v
```

### Breakdown of Test Results:
- **`tests/test_preprocessing.py` (15/15 Passed):**
  - Image loading: RGB optical, Sentinel-2 13-band, Sentinel-1 2-band SAR, missing/empty file checks.
  - GeoTIFF metadata: Tiepoint, scale, EPSG:4326 extraction, CRS compatibility checking, disjoint scene rejection.
  - Sub-pixel alignment: 2x3 affine matrix derivation, synthetic forward warping, SpaceNet 8 end-to-end alignment.
  - Diagnostics: Checkerboard generation, alpha overlay blend, difference map, and report JSON export.
- **`tests/test_detection.py` (15/15 Passed):**
  - Zero change on identical images.
  - Known rectangle detection accuracy.
  - Strict exclusion of differences in invalid mask regions.
  - Error handling for missing, 0-byte, and dimensionally incompatible inputs.
  - Zero-division safety with 0 valid pixels.
  - Return contract schema and dataclass verification.
- **`tests/test_app.py` (18/18 Passed):**
  - `GET /api/health` status, versions, sample availability.
  - `GET /api/sample-datasets` metadata listing.
  - `POST /api/detect/sample` baseline retrieval and dynamic custom thresholds ($t=60$, `color_diff`).
  - Validation guards: negative threshold rejection, invalid method rejection, non-existent sample 404.
  - `POST /api/detect/upload` synthetic image pairs, dimension mismatch rejection, corrupt file rejection, 0-byte file rejection.
  - Overlap mask invalid pixel exclusion during upload analysis.
  - Artifact serving with security guards against directory traversal.
  - Mandatory preservation of scientific disclaimer text.
  - Web dashboard HTML delivery (`GET /`).

**Total Result:** `Ran 48 tests in 0.956s — OK (48 passed, 0 failed, 0 errors)`.

---

## 7. How to Run the Complete Application

1. From the project directory (`FloodScope_AI`), start the server:
   ```powershell
   python scripts/run_app.py
   ```
   *(Or `py scripts/run_app.py`)*
2. Open your web browser to:
   ```text
   http://127.0.0.1:8000
   ```
3. Use the interactive controls:
   - Inspect the SpaceNet 8 Germany sample tile across all 5 visualizer panes (Overlay, Mask, Diff, Pre/Post Side-by-Side, Alignment Diagnostics).
   - Adjust the threshold slider or click the quick presets (25, 35, 50, 75).
   - Switch between grayscale luminance delta and per-band color distance.
   - Upload custom paired satellite scenes for automated candidate surface change evaluation.

---

## 8. Remaining Blockers or Manual Steps

- **No Technical Blockers:** The complete end-to-end stack (preprocessing, detection baseline, FastAPI backend, web interface, test suite) is functional and verified.
- **Note on Git Status:** The directory is an unzipped project folder without `.git`. If the user wishes to push this repository to GitHub or GitLab, a simple `git init`, `git add .`, and remote configuration can be executed when authorized.

---

## 9. Scientific Limitations & Responsible Interpretation

> [!IMPORTANT]
> **Summary of Domain Constraints:**
> - The OpenCV baseline calculates mathematical pixel intensity variations. It cannot determine water depth, chemical inundation signatures, or hydrological continuity.
> - In the SpaceNet 8 sample tile (`0_16_67`), significant surface changes stem from summer deciduous tree canopy development, crop cultivation, and cloud shadows.
> - The SpaceNet 8 sample GeoJSON annotations represent building footprints and road centerlines, **not bi-temporal water inundation ground truth**.
> - Detection metrics must always be reported as **Candidate Surface Changes** or **Observed Radiometric Deltas**, never as confirmed flood probability or flood extent percentage.
