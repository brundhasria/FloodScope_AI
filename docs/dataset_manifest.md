# FloodScope AI — Dataset Manifest & Specification

## 1. Overview & Source Information
- **Dataset Name:** Sen1Floods11 (v1.1)
- **Primary Source / Repository:** [Cloud to Street (Floodbase) Sen1Floods11 GitHub](https://github.com/cloudtostreet/Sen1Floods11)
- **Publication Reference:** Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* IEEE/CVF CVPR Workshops 2020, pp. 210-211.
- **Remote Storage Location:** Google Cloud Storage (`gs://sen1floods11/v1.1/`)
- **Direct HTTPS Endpoint:** `https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2F...`

## 2. Licensing & Distribution
- **Repository License:** The official GitHub repository does not include an explicit standalone `LICENSE` file. The paper and dataset were published under Open Access (CVPRW 2020) for academic and benchmarking purposes. Mirror distributions (e.g., Hugging Face, Kaggle) cite attribution under **CC BY 4.0**.
- **Repository Limits & Git Best Practices:**
  - GitHub enforces a strict 100 MB single-file limit and recommends keeping total repository size under 2 GB.
  - The complete Sen1Floods11 dataset exceeds 14 GB (and up to ~30 GB with weak annotations).
  - **Rule:** Never commit raw or processed `.tif` files to version control. All raster files must remain strictly within local `dataset/raw/`, `dataset/processed/`, and `dataset/labels/` and are ignored via `.gitignore`.

## 3. Dataset Architecture & Modalities

| Modality / Split | Directory in GCS (`v1.1/data/`) | Count | Format | Dimensions | Band Information |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sentinel-1 Hand-Labeled** | `flood_events/HandLabeled/S1Hand/` | 446 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 2 bands: Band 1 = VV polarization, Band 2 = VH polarization (10m resolution) |
| **Sentinel-2 Hand-Labeled** | `flood_events/HandLabeled/S2Hand/` | 446 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 13 optical bands (L1C top-of-atmosphere reflectance) |
| **Ground Truth (Hand)** | `flood_events/HandLabeled/LabelHand/` | 446 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 1 band: `1` = Water, `0` = Land / Non-water, `-1` = Excluded / NoData |
| **JRC Permanent Water** | `flood_events/HandLabeled/JRCWaterHand/` | 446 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 1 band: `1` = Permanent water, `0` = Land |
| **Otsu Baseline Labels** | `flood_events/HandLabeled/S1OtsuLabelHand/` | 446 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 1 band: `1` = Otsu detected water, `0` = Land |
| **Weakly Labeled S1** | `flood_events/WeaklyLabeled/S1Weak/` | 4,385 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 2 bands: VV, VH |
| **Weakly Labeled S2 Index** | `flood_events/WeaklyLabeled/S2IndexLabelWeak/`| 4,385 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | 1 band: Optical thresholding (MNDWI/NDWI) |
| **Permanent Water Chips** | `perm_water/S1Perm/` & `JRCPerm/` | 815 | Cloud-Optimized GeoTIFF (`.tif`) | 512 × 512 | SAR and permanent water reference masks |

## 4. Critical Assessment: Before-and-After Flood Imagery
> [!WARNING]
> **Sen1Floods11 is a UNI-TEMPORAL dataset.**
> - It captures satellite imagery **during / immediately following** 11 global flood events.
> - It **does NOT** provide matched pre-flood (before) optical or SAR satellite images out of the box.
> - The included `JRCWaterHand` layer is a static historical surface water mask (showing where water normally exists year-round), **not** a pre-flood satellite image capture.
>
> **Project Impact for FloodScope AI:**
> 1. For flood surface segmentation training (Member 2), Sen1Floods11 hand-labeled tiles provide excellent SAR/optical ground truth.
> 2. For the before-and-after change comparison feature, pre-flood scenes must be acquired via bounding boxes in `Sen1Floods11_Metadata.geojson` (e.g. via Copernicus Open Access / Google Earth Engine) or complemented with a bi-temporal flood change dataset (such as SpaceNet 8 or Copernicus EMS activations).

## 5. Local Dataset Inventory
*Status as of sample ingestion:*
- **Total Local Files:** 6 files (4 raw imagery, 2 ground-truth labels)
- **Verified Chips:** `Bolivia_103757`, `Bolivia_129334`

| File Name | Folder | File Size | Dimensions | Bands | Data Type | Value Range / Unique Values | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Bolivia_103757_S1Hand.tif` | `dataset/raw/` | 698,748 B (0.67 MB) | 512 × 512 | 2 (VV, VH) | float32 | [-40.82 dB, -0.69 dB] | Verified Readable |
| `Bolivia_103757_S2Hand.tif` | `dataset/raw/` | 908,533 B (0.87 MB) | 512 × 512 | 13 (Optical) | int32 | [0.0, 4596.0] | Verified Readable |
| `Bolivia_103757_LabelHand.tif`| `dataset/labels/`| 7,609 B (0.01 MB) | 512 × 512 | 1 (Mask) | int32 | Unique: `[-1, 0, 1]` | Verified Ground Truth |
| `Bolivia_129334_S1Hand.tif` | `dataset/raw/` | 1,764,952 B (1.68 MB) | 512 × 512 | 2 (VV, VH) | float32 | [-36.83 dB, 0.47 dB] | Verified Readable |
| `Bolivia_129334_S2Hand.tif` | `dataset/raw/` | 2,191,905 B (2.09 MB) | 512 × 512 | 13 (Optical) | int32 | [1171.0, 3957.0] | Verified Readable |
| `Bolivia_129334_LabelHand.tif`| `dataset/labels/`| 13,085 B (0.01 MB) | 512 × 512 | 1 (Mask) | int32 | Unique: `[-1, 0, 1]` | Verified Ground Truth |
| `SpaceNet8_Germany_0_16_67_PRE.tif` | `dataset/raw/` | 3,330,496 B (3.18 MB) | 1300 × 1300 | 3 (RGB) | uint8 | [0, 255] | Verified Readable (Pre-Event) |
| `SpaceNet8_Germany_0_16_67_POST.tif`| `dataset/raw/` | 9,124,983 B (8.70 MB) | 1743 × 1743 | 3 (RGB) | uint8 | [0, 255] | Verified Readable (Post-Event) |
| `SpaceNet8_Germany_0_16_67_annotations.geojson` | `dataset/labels/` | 8,591 B (8.39 KB) | Vector | N/A | GeoJSON | 16 features (8 lines, 8 polys) | Verified Ground Truth |

## 6. Bi-Temporal (Before & After) Imagery Investigation

To satisfy FloodScope AI's before-and-after flood comparison requirement, three potential sources for genuine pre-event and post-event imagery were evaluated against constraints of a student project (one-week timeline, low disk storage, zero cloud fees):

| Evaluation Criteria | Option A: SpaceNet 8 (Flood Challenge) | Option B: Copernicus EMS (Rapid Mapping) | Option C: Custom Copernicus CDSE (Sentinel-1/2 Archives) |
| :--- | :--- | :--- | :--- |
| **Real Pre- & Post-Event Imagery?** | **YES.** Natively provides paired pre-flood and post-flood optical captures of the identical geographic tiles. | **NO (Vector only).** Delivers vector delineations and PDF maps; raw underlying imagery is commercial and restricted. | **YES (Manual).** Pre-flood and post-flood Sentinel scenes exist in open ESA archives. |
| **Spatial & Grid Alignment** | **Geographically Aligned.** Pre- and post-event tiles share identical geographic bounds in EPSG:4326 (>99.98% bounding overlap). Note: Different native pixel resolutions (1300×1300 vs 1743×1743). | N/A (Raster tiles not provided directly). | **Uncalibrated.** Requires manual co-registration, orthorectification, and coordinate clipping to 512x512. |
| **Licensing** | **CC BY-SA 4.0** (Open commercial & academic use with attribution). | EU Open Access (Derivative maps open; imagery restricted). | Open Access (Copernicus Sentinel Terms). |
| **Download Accessibility** | Open AWS S3 HTTP (`https://spacenet-dataset.s3.amazonaws.com/...`). Supports single-tile HTTP downloads without AWS keys or CLI tools. | Requires portal registration; GIS shapefiles/GPKG downloads. | Requires CDSE OAuth2 credentials and heavy REST queries. |
| **Storage Footprint** | Minimal: **~12.46 MB** for a complete paired sample set (`PRE` + `POST` + `annotations`). | Moderate: Vector packages are 50–200 MB, but contain zero raster images. | Very Heavy: Raw Sentinel `.SAFE` scenes are ~1 GB per timestamp (~2 GB unzipped per pair). |
| **Feasibility for 1-Week Schedule**| **High / Recommended.** Drop-in ready for Member 1 verification and Member 2/3 change detection. | **Low.** Does not solve the satellite raster comparison requirement. | **Low.** Complex SAR/optical processing chain (SNAP/GDAL) takes several days to build and calibrate. |

### Key Distinction: Real Imagery vs Reference Masks
* **Real Before-and-After Imagery (SpaceNet 8):** Genuinely distinct optical sensor observations acquired on two separate dates over the same physical coordinates (e.g., sunny pre-disaster landscape vs inundated post-disaster floodwaters).
* **Static Water Masks (Sen1Floods11 `JRCWaterHand`):** A historical synthetic map showing where water was detected over a 35-year span. It does not reflect atmospheric, vegetation, or seasonal conditions immediately prior to the flood event.

## 7. Verified SpaceNet 8 Geospatial Compatibility Audit (Rasterio)

A rigorous geospatial inspection was executed using `rasterio` on the downloaded tile pair (`0_16_67`):

| Parameter | Pre-Event Tile (`_PRE.tif`) | Post-Event Tile (`_POST.tif`) | Compatibility Assessment |
| :--- | :--- | :--- | :--- |
| **File Identifier** | `10500500C4DD7000_0_16_67` | `1040050035DC3B00_0_16_67` | Same tile coordinate (`0_16_67`) in official mapping CSV |
| **Pixel Dimensions** | **1300 × 1300** | **1743 × 1743** | **Grid Mismatch (Expected):** Post-event capture had higher sensor sampling rate. Requires bilinear resampling before direct pixel subtraction. |
| **Bands & Dtype** | 3 bands (RGB), `uint8` | 3 bands (RGB), `uint8` | **Matched:** Standard optical 8-bit channels. |
| **CRS** | `EPSG:4326` (WGS 84) | `EPSG:4326` (WGS 84) | **Identical CRS.** |
| **Geographic Bounds** | Left: `6.9177181°`<br>Bottom: `50.4872861°`<br>Right: `6.9225840°`<br>Top: `50.4921520°` | Left: `6.9177169°`<br>Bottom: `50.4872849°`<br>Right: `6.9225864°`<br>Top: `50.4921544°` | **>99.98% Geographic Overlap:** Coordinates differ by < 0.000002° (~15 cm). Ground footprint is identical. |
| **Pixel Resolution** | `3.74298e-6` deg (~0.35m) | `2.79376e-6` deg (~0.28m) | Ground resolution is slightly finer in post-event image. |
| **Annotations CRS** | `urn:ogc:def:crs:OGC:1.3:CRS84` | 16 features (8 `Polygon` buildings, 8 `LineString` roads) | GeoJSON coordinates directly match `EPSG:4326`. |


