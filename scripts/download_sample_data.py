"""
FloodScope AI - Sample Dataset Ingestion Script
Member 1: Dataset Preparation and Verification

Safely downloads a verified sample of Sen1Floods11 tiles from the public GCS endpoint
directly over HTTPS, without requiring gsutil or cloud authentication credentials.
"""

import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

BASE_GCS_URL = "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/"

# Curated sample chips across distinct flood events
SAMPLE_CHIPS = [
    ("Bolivia_103757", "Bolivia"),
    ("Bolivia_129334", "Bolivia"),
    ("USA_430764", "USA"),
    ("Mekong_1117585", "Mekong"),
    ("Somalia_1044439", "Somalia"),
]

MODALITY_PATHS = [
    # (Remote path template, Local target directory, Remote suffix)
    ("v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FS1Hand%2F{chip}_S1Hand.tif", "dataset/raw", "_S1Hand.tif"),
    ("v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FS2Hand%2F{chip}_S2Hand.tif", "dataset/raw", "_S2Hand.tif"),
    ("v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FLabelHand%2F{chip}_LabelHand.tif", "dataset/labels", "_LabelHand.tif"),
    ("v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FJRCWaterHand%2F{chip}_JRCWaterHand.tif", "dataset/labels", "_JRCWaterHand.tif"),
]


def download_sample(base_dir: Path, max_chips: int = 2) -> None:
    """Downloads a minimal, verified sample set for initial pipeline verification."""
    selected = SAMPLE_CHIPS[:max_chips]
    print(f"Downloading {len(selected)} sample chip sets from Sen1Floods11 (v1.1)...")

    for chip_id, event in selected:
        print(f"\nProcessing Chip: {chip_id} (Event: {event})")
        for remote_tmpl, local_dir_str, suffix in MODALITY_PATHS:
            object_name = remote_tmpl.format(chip=chip_id)
            url = f"{BASE_GCS_URL}{object_name}?alt=media"
            
            local_dir = base_dir / local_dir_str
            local_dir.mkdir(parents=True, exist_ok=True)
            local_file = local_dir / f"{chip_id}{suffix}"

            if local_file.exists():
                print(f"  [SKIPPED] {local_file.name} already exists.")
                continue

            print(f"  Downloading {local_file.name} ... ", end="", flush=True)
            try:
                urllib.request.urlretrieve(url, local_file)
                size_mb = local_file.stat().st_size / (1024 * 1024)
                print(f"DONE ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"FAILED ({e})")
                if local_file.exists():
                    local_file.unlink()

    print("\nDownload batch finished.")
    print("Run `python scripts/verify_dataset.py` to audit downloaded assets.")


if __name__ == "__main__":
    download_sample(Path("."), max_chips=2)
