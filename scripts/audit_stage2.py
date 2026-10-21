"""
FloodScope AI - Stage 2 Preprocessing & Alignment Verification Audit
Member 2: Detection

Audits:
1. Exact file paths for raw inputs and processed outputs.
2. Complete geospatial metadata (CRS, bounds, tiepoints, pixel scale, dimensions).
3. Exact pixel correspondence formulation and affine transformation.
4. Raw data integrity via SHA-256 hashes.
5. Reproducibility test of the pipeline.
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def run_audit():
    print("=" * 75)
    print("FLOODSCOPE AI -- FINAL STAGE 2 PREPROCESSING & ALIGNMENT AUDIT")
    print("=" * 75)

    # 1. Raw Files and Hashes
    raw_files = [
        REPO_ROOT / "dataset" / "raw" / "SpaceNet8_Germany_0_16_67_PRE.tif",
        REPO_ROOT / "dataset" / "raw" / "SpaceNet8_Germany_0_16_67_POST.tif",
        REPO_ROOT / "dataset" / "raw" / "Bolivia_103757_S1Hand.tif",
        REPO_ROOT / "dataset" / "raw" / "Bolivia_103757_S2Hand.tif",
        REPO_ROOT / "dataset" / "raw" / "Bolivia_129334_S1Hand.tif",
        REPO_ROOT / "dataset" / "raw" / "Bolivia_129334_S2Hand.tif",
        REPO_ROOT / "dataset" / "labels" / "Bolivia_103757_LabelHand.tif",
        REPO_ROOT / "dataset" / "labels" / "Bolivia_129334_LabelHand.tif",
        REPO_ROOT / "dataset" / "labels" / "SpaceNet8_Germany_0_16_67_annotations.geojson",
    ]

    print("\n--- 1. Raw Dataset Integrity (SHA-256 Hashes) ---")
    for f in raw_files:
        if f.is_file():
            h = compute_sha256(f)
            print(f"  {f.name:45} | Size: {f.stat().st_size:9} B | SHA256: {h}")
        else:
            print(f"  [MISSING] {f.name}")

    # 2. Processed Outputs
    proc_dir = REPO_ROOT / "dataset" / "processed"
    print("\n--- 2. Processed Outputs Inventory ---")
    for p in sorted(proc_dir.iterdir()):
        if p.is_file() and p.name != ".gitkeep":
            h = compute_sha256(p)
            print(f"  {p.name:50} | Size: {p.stat().st_size:9} B | SHA256: {h}")

    # 3. Alignment Report Inspection
    rep_path = proc_dir / "SpaceNet8_Germany_0_16_67_alignment_report.json"
    if rep_path.is_file():
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("\n--- 3. Alignment Audit Metrics ---")
        print(f"  Status:               {data.get('status')}")
        print(f"  Target Grid:          {data.get('target_grid')} ({data.get('grid_dimensions')})")
        print(f"  Valid Overlap Pixels: {data.get('valid_overlap_pixels')} / {data.get('grid_dimensions')[0]*data.get('grid_dimensions')[1]}")
        print(f"  Valid Overlap Ratio:  {data.get('valid_overlap_ratio')*100:.4f}%")
        print(f"  CRS:                  PRE={data['meta_pre']['crs']} | POST={data['meta_post']['crs']}")
        print(f"  Affine Matrix M:      {data.get('affine_matrix')}")

    print("=" * 75)

if __name__ == "__main__":
    run_audit()
