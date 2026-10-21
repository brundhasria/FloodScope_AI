# FloodScope AI — Member 1 Dataset Handover Checklist

This checklist tracks dataset preparation and verification milestones before handing off assets to Member 2 (Model Training & Baseline Development).

---

### Phase 1: Directory Setup & Architecture
- [x] **Project Folder Hierarchy Created:**
  - `dataset/raw/`
  - `dataset/processed/`
  - `dataset/labels/`
  - `docs/`
  - `scripts/`
  - `README.md`
- [x] **Git Protection Enforced:** `.gitignore` configured to prevent large GeoTIFF files (`*.tif`) and dataset folders from being committed to GitHub.

---

### Phase 2: Dataset Due Diligence & Technical Analysis
- [x] **Repository & Source Inspected:** Official repository analyzed: [cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11).
- [x] **Licensing Status Documented:** Academic / open research use (often cited under CC BY 4.0; no explicit GitHub LICENSE file; commercial redistribution restrictions noted).
- [x] **Format & Metadata Documented:**
  - Format: Cloud-Optimized GeoTIFF (`.tif`), 512 × 512 chips, 10m spatial resolution.
  - S1: 2 bands (VV, VH).
  - S2: 13 optical bands (L1C).
  - Labels: `1` = Water, `0` = Land, `-1` = Excluded/NoData.
- [x] **Temporal Alignment Verified (Before-and-After Check):**
  - **FINDING:** Sen1Floods11 is **uni-temporal** (during-flood event only).
  - It does **not** provide pre-flood matched images out-of-the-box (only static JRC permanent water reference).
  - Project strategy documented for before/after comparison feature.

---

### Phase 3: Manifest & Verification Tools
- [x] **Dataset Manifest Created:** [dataset_manifest.md](file:///c:/Users/sreed/Downloads/FloodScope_AI/docs/dataset_manifest.md).
- [x] **Python Verification Script Created:** [verify_dataset.py](file:///c:/Users/sreed/Downloads/FloodScope_AI/scripts/verify_dataset.py).
- [x] **Zero-File Baseline Execution Tested:** Tested against empty folders; reported 0 files truthfully without fabrication.

---

### Phase 4: Data Ingestion & Audit (Sample Set Verified)
- [x] **Sample Ingestion Approved:** User confirmed downloading minimal 2-chip sample set (`Bolivia_103757`, `Bolivia_129334`).
- [x] **Physical Integrity Verified:** All 6 Sen1Floods11 `.tif` files successfully read by Pillow/NumPy; dimensions confirmed to be exactly 512 × 512; bands match specifications (S1 = 2 bands, S2 = 13 bands, Labels = 1 band).
- [x] **Label Consistency Checked:** Unique label values verified to contain `[-1, 0, 1]`.
- [x] **Raw vs Label Pairing Confirmed:** 100% pairing match between downloaded raw images and corresponding labels (0 unpaired).

---

### Phase 4.5: Bi-Temporal Imagery Audit (SpaceNet 8 Verified)
- [x] **Tile Pairing Verified:** Official mapping CSV verified (`0_16_67.geojson` maps to `10500500C4DD7000_0_16_67.tif` and `1040050035DC3B00_0_16_67.tif`).
- [x] **Targeted Ingestion Executed:** Pre-event, post-event GeoTIFFs and GeoJSON annotations acquired without modifying Sen1Floods11 samples.
- [x] **Geospatial Audit Run (Rasterio):**
  - CRS verified: Both PRE and POST share `EPSG:4326`.
  - Geographic bounds verified: >99.98% bounding overlap.
  - Pixel grid mismatch identified: 1300×1300 vs 1743×1743 documented as a required resampling step for Member 2.
- [x] **Vector Annotations Inspected:** 16 features (8 `Polygon` buildings, 8 `LineString` roads) verified in `OGC:CRS84`.

---

### Phase 5: Member 1 Handover Sign-Off to Member 2
- [x] **Dataset Manifest Complete:** Full inventory, licensing, format specifications, and verification logs documented in `docs/dataset_manifest.md`.
- [x] **Verification Tooling Operational:** `scripts/verify_dataset.py` audits all 9 local files with 0 corruptions and 100% pairing.
- [x] **Technical Briefing for Member 2:**
  - **Module A (Flood Segmentation):** Use `dataset/raw/Bolivia_*` and `dataset/labels/Bolivia_*` (Sen1Floods11) for SAR/optical water mask learning.
  - **Module B (Before-and-After Comparison):** Use `dataset/raw/SpaceNet8_Germany_0_16_67_PRE.tif` and `POST.tif` for change detection, applying spatial resampling as documented.
- [x] **Member 1 Sign-Off:** Complete and verified. Ready for Member 2 handoff.

