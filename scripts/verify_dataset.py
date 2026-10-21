"""
FloodScope AI - Dataset Verification & Health Check Script
Member 1: Dataset Preparation and Verification

This script inspects the local dataset directories, verifies the physical integrity
of satellite images and labels, extracts metadata (dimensions, channels, value distributions),
and checks pairing between raw imagery and ground-truth labels.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from PIL import Image
    import numpy as np
except ImportError as exc:
    print(f"Error: Missing required packages: {exc}")
    print("Please install requirements: pip install pillow numpy")
    sys.exit(1)


def inspect_file(file_path: Path) -> Dict[str, Any]:
    """Inspects a raster image or GeoJSON vector file for validity and metadata."""
    info = {
        "file_name": file_path.name,
        "relative_path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "is_readable": False,
        "error": None,
        "dimensions": None,
        "bands": 0,
        "dtype": None,
        "unique_values": None,
        "min_val": None,
        "max_val": None,
        "format": None,
        "details": None,
    }

    # Handle GeoJSON Vector Annotations
    if file_path.suffix.lower() in [".geojson", ".json"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            info["is_readable"] = True
            info["format"] = "GeoJSON"
            features = data.get("features", [])
            geom_types = {}
            for ft in features:
                gt = ft.get("geometry", {}).get("type", "Unknown")
                geom_types[gt] = geom_types.get(gt, 0) + 1
            info["details"] = f"Features: {len(features)}, Types: {geom_types}"
        except Exception as e:
            info["error"] = f"Invalid JSON/GeoJSON: {e}"
        return info

    # Handle Raster Images (GeoTIFF, TIFF, PNG, etc.)
    try:
        with Image.open(file_path) as img:
            info["is_readable"] = True
            info["format"] = img.format
            width, height = img.size
            
            # Check band count via TIFF Tag 277 (SamplesPerPixel), n_frames, or getbands()
            bands = 1
            if hasattr(img, "tag_v2") and 277 in img.tag_v2:
                bands = int(img.tag_v2[277])
            elif hasattr(img, "n_frames") and img.n_frames > 1:
                bands = img.n_frames
            elif len(img.getbands()) > 1:
                bands = len(img.getbands())

            info["dimensions"] = (height, width)
            info["bands"] = bands

            # Read array data to verify corrupted bytes and check label values
            arr = np.array(img)
            info["dtype"] = str(arr.dtype)
            
            # Compute finite min/max (S1 chips can have NaN in no-data border regions)
            finite_mask = np.isfinite(arr)
            if np.any(finite_mask):
                info["min_val"] = float(np.min(arr[finite_mask]))
                info["max_val"] = float(np.max(arr[finite_mask]))

            # For labels or categorical masks, record unique values
            unique_vals = np.unique(arr[finite_mask]) if np.any(finite_mask) else np.unique(arr)
            if len(unique_vals) <= 16:
                info["unique_values"] = [int(v) if np.issubdtype(arr.dtype, np.integer) else float(v) for v in unique_vals]

    except Exception as e:
        info["error"] = str(e)

    return info


def verify_dataset(base_dir: Path) -> Dict[str, Any]:
    """Scans dataset directories and runs verification checks."""
    raw_dir = base_dir / "dataset" / "raw"
    processed_dir = base_dir / "dataset" / "processed"
    labels_dir = base_dir / "dataset" / "labels"

    print("=" * 70)
    print("FloodScope AI -- Dataset Verification Audit")
    print("=" * 70)
    print(f"Project root: {base_dir.resolve()}")
    print(f"Raw directory:       {raw_dir} (exists: {raw_dir.is_dir()})")
    print(f"Processed directory: {processed_dir} (exists: {processed_dir.is_dir()})")
    print(f"Labels directory:    {labels_dir} (exists: {labels_dir.is_dir()})")
    print("-" * 70)

    # Collect supported raster files (excluding git keeps / temp files)
    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".geojson", ".json"}
    
    def get_raster_files(directory: Path) -> List[Path]:
        if not directory.is_dir():
            return []
        return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts]

    raw_files = get_raster_files(raw_dir)
    processed_files = get_raster_files(processed_dir)
    label_files = get_raster_files(labels_dir)

    print(f"Discovered Files:")
    print(f"  - dataset/raw/       : {len(raw_files)} files")
    print(f"  - dataset/processed/ : {len(processed_files)} files")
    print(f"  - dataset/labels/    : {len(label_files)} files")
    print("-" * 70)

    total_files = len(raw_files) + len(processed_files) + len(label_files)
    if total_files == 0:
        print("[STATUS] No dataset image files found in the dataset directories.")
        print("         Directories are initialized and ready for sample/full ingestion.")
        print("=" * 70)
        return {
            "total_files": 0,
            "readable_count": 0,
            "corrupt_count": 0,
            "paired_count": 0,
            "unpaired_count": 0,
            "files": []
        }

    # Inspect all files
    all_inspections = []
    unreadable = []
    for f in raw_files + processed_files + label_files:
        info = inspect_file(f)
        all_inspections.append(info)
        if not info["is_readable"]:
            unreadable.append(info)
            print(f"[CORRUPT/UNREADABLE] {f.name}: {info['error']}")
        elif info.get("format") == "GeoJSON":
            print(f"[OK] {f.name} | Format: GeoJSON | {info['details']}")
        else:
            dim_str = f"{info['dimensions'][0]}x{info['dimensions'][1]}" if info['dimensions'] else "N/A"
            vals_str = f"unique values={info['unique_values']}" if info['unique_values'] is not None else f"range=[{info['min_val']}, {info['max_val']}]"
            print(f"[OK] {f.name} | Dims: {dim_str} | Bands: {info['bands']} | Type: {info['dtype']} | {vals_str}")

    # Check Raw vs Label Pairing
    print("-" * 70)
    print("Checking Raw Imagery <-> Label Pairing:")
    # Extract IDs (e.g. Bolivia_103757)
    def extract_stem(p: Path) -> str:
        name = p.stem
        for suffix in ["_S1Hand", "_S2Hand", "_LabelHand", "_S1OtsuLabelHand", "_JRCWaterHand", "_PRE", "_POST", "_annotations"]:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name

    raw_stems = {extract_stem(p): p for p in raw_files}
    label_stems = {extract_stem(p): p for p in label_files}

    paired_stems = set(raw_stems.keys()).intersection(set(label_stems.keys()))
    unpaired_labels = set(label_stems.keys()) - set(raw_stems.keys())
    unpaired_raw = set(raw_stems.keys()) - set(label_stems.keys())

    print(f"  - Paired (Raw + Label)     : {len(paired_stems)}")
    print(f"  - Unpaired Labels (Missing Raw): {len(unpaired_labels)}")
    print(f"  - Unpaired Raw (Missing Label) : {len(unpaired_raw)}")

    if unpaired_labels:
        print(f"    Missing raw images for: {list(unpaired_labels)[:5]}")
    if unpaired_raw:
        print(f"    Missing labels for: {list(unpaired_raw)[:5]}")

    print("=" * 70)
    print(f"Verification Summary:")
    print(f"  Total Files Verified : {total_files}")
    print(f"  Valid & Readable     : {total_files - len(unreadable)}")
    print(f"  Unreadable / Corrupt : {len(unreadable)}")
    print("=" * 70)

    return {
        "total_files": total_files,
        "readable_count": total_files - len(unreadable),
        "corrupt_count": len(unreadable),
        "paired_count": len(paired_stems),
        "unpaired_count": len(unpaired_labels) + len(unpaired_raw),
        "inspections": all_inspections
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify FloodScope AI dataset integrity.")
    parser.add_argument("--base-dir", type=str, default=".", help="Project root directory containing dataset/")
    args = parser.parse_args()

    verify_dataset(Path(args.base_dir))
