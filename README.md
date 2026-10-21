# FloodScope AI

FloodScope AI is an automated satellite imagery analysis application designed to detect candidate surface changes and evaluate bi-temporal radiometric deltas in remote sensing imagery.

## Project Structure
```text
FloodScope-AI/
├── dataset/
│   ├── raw/         # Raw downloaded satellite imagery and external masks
│   ├── processed/   # Preprocessed, aligned scenes and baseline change products
│   └── labels/      # Ground-truth water masks and vector annotations
├── docs/            # Dataset specifications, manifests, and technical notes
├── scripts/         # Dataset verification, pipeline execution, and app launcher
├── src/
│   └── floodscope/  # Core package: preprocessing, detection baseline, and web app
├── tests/           # Automated test suite (preprocessing, detection, and API tests)
├── .gitignore       # Git ignore rules for heavy remote sensing assets
└── README.md        # Project overview and operational guide
```

## Team Responsibilities & Status
- **Member 1:** Dataset acquisition, verification, manifest generation, and quality auditing. *(Complete)*
- **Member 2:** Multi-sensor preprocessing, sub-pixel geospatial alignment, and OpenCV candidate change detection baseline. *(Complete)*
- **Member 3:** FastAPI web backend, interactive multi-view visualization dashboard, upload workflows, and integration verification. *(Complete)*

---

## Prerequisites & Dependencies

The project is tested and verified under:
- **Python:** `3.14.3` (or `3.13.14`)
- **OpenCV:** `5.0.0` (`opencv-python 5.0.0.93`)
- **NumPy:** `2.5.0`
- **Tifffile:** `2026.6.1`
- **FastAPI:** `0.111.0`
- **Uvicorn:** `0.30.1`
- **Starlette / TestClient:** `0.37.2`

No third-party test runners or heavy C-extensions (like GDAL) are required. Standard library `unittest` executes all test suites.

---

## How to Run the Application

Start the web dashboard and API server:
```powershell
python scripts/run_app.py
```
*(On Windows systems where `python` is aliased, run `py scripts/run_app.py`)*

Then open your browser to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Dashboard Features:
1. **Verified Sample Mode:** Instant interactive inspection of the SpaceNet 8 Germany (`0_16_67`) bi-temporal satellite pair.
2. **Interactive Parameter Controls:** Adjust difference threshold $\Delta$ ($0\text{--}255$), toggle between grayscale luminance difference and per-band RGB Euclidean distance, and toggle morphological opening noise filtering.
3. **Custom Image Upload:** Upload any matched before/after satellite image pair (PNG/JPEG/TIFF up to 25MB) with optional overlap validity mask.
4. **Multi-View Visualizer:**
   - **Candidate Change Overlay:** 50% alpha red highlight of detected surface changes on reference imagery.
   - **Candidate Change Mask:** High-contrast binary mask ($255=\text{change}$, $0=\text{no change/invalid}$).
   - **Difference Image:** Raw intensity delta map.
   - **Side-by-Side:** Pre-event vs post-event capture viewer.
   - **Spatial Alignment Diagnostics:** Checkerboard composite (130px squares), 50/50 dual-scene blend, and valid overlap footprint mask.
5. **Quantitative Summary:** Total pixels, valid footprint pixels, candidate changed pixels, and **Change % Within Valid Area**.

---

## How to Run the Processing Pipeline

To regenerate alignment and baseline detection artifacts directly via command-line:
```powershell
# 1. Run geospatial alignment
python scripts/run_alignment.py

# 2. Run multi-threshold change detection baseline
python scripts/run_detection.py

# 3. Verify processed output integrity
python scripts/verify_processed_outputs.py
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves interactive web dashboard. |
| `GET` | `/api/health` | Health check, OpenCV/Python versions, and sample availability. |
| `GET` | `/api/sample-datasets` | Lists available sample datasets and pre-computed baseline artifact URLs. |
| `POST` | `/api/detect/sample` | Runs candidate detection on sample pair with custom `threshold`, `method`, `apply_morphology`. |
| `POST` | `/api/detect/upload` | Accepts uploaded Before and After images (`multipart/form-data`) and runs detection. |
| `GET` | `/api/outputs/{run_id}/{filename}` | Securely serves generated visual artifacts with path traversal protections. |

---

## How to Run the Automated Test Suite

Run the unified test suite across all project modules:
```powershell
python -m unittest discover tests -v
```
*(Or `py -m unittest discover tests -v`)*

**Actual Result:** **48/48 tests passed (0 failures, 0 errors)**:
- 15 Preprocessing & Alignment tests (`tests/test_preprocessing.py`)
- 15 Detection Baseline tests (`tests/test_detection.py`)
- 18 Application Backend & Integration tests (`tests/test_app.py`)

---

## Storage of Generated Artifacts

- **Pre-computed Baseline:** Stored under `dataset/processed/` with prefix `SpaceNet8_Germany_0_16_67_*`.
- **Dynamic & Upload Runs:** Stored under `dataset/processed/runs/{run_id}/`.
- **Integrity Guarantee:** Original raw files under `dataset/raw/` and labels under `dataset/labels/` are never overwritten or modified.

---

## Critical Scientific Disclaimers

> [!WARNING]
> **Candidate Surface Changes $\neq$ Confirmed Flooding**
>
> 1. Detections generated by this pipeline represent generic pixel-level radiometric or intensity deltas between satellite captures.
> 2. Primary real-world causes of false positives include:
>    - Seasonal vegetation growth and phenology (e.g., bare trees in spring vs dense summer canopy).
>    - Agricultural operations (tilled soil, harvest cycles).
>    - Atmospheric cloud cover, low cloud banks, and cast shadows.
>    - Variations in solar zenith angle and acquisition geometry.
> 3. The current SpaceNet 8 sample tile is **unbenchmarked against independent bi-temporal floodwater ground truth**.
> 4. `valid_area_change_pct` reflects observed image differences within the valid geographic footprint—**never present this value as the percentage of land flooded**.
