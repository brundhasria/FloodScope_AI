"""
FloodScope AI - Targeted 2-Chip Sample Downloader
Member 1: Dataset Preparation and Verification

Downloads verified Sentinel-1, Sentinel-2, and ground-truth label GeoTIFFs
for chips Bolivia_103757 and Bolivia_129334 from the official Sen1Floods11 bucket.
"""

import os
import sys
import urllib.request
from pathlib import Path

TARGETS = [
    {
        "chip": "Bolivia_103757",
        "type": "Sentinel-1",
        "url": "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FS1Hand%2FBolivia_103757_S1Hand.tif?generation=1601572653484826&alt=media",
        "dest": Path("dataset/raw/Bolivia_103757_S1Hand.tif"),
        "expected_bytes": 698748,
    },
    {
        "chip": "Bolivia_103757",
        "type": "Sentinel-2",
        "url": "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FS2Hand%2FBolivia_103757_S2Hand.tif?generation=1601573197287337&alt=media",
        "dest": Path("dataset/raw/Bolivia_103757_S2Hand.tif"),
        "expected_bytes": 908533,
    },
    {
        "chip": "Bolivia_103757",
        "type": "LabelHand",
        "url": "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FLabelHand%2FBolivia_103757_LabelHand.tif?generation=1601572621199497&alt=media",
        "dest": Path("dataset/labels/Bolivia_103757_LabelHand.tif"),
        "expected_bytes": 7609,
    },
    {
        "chip": "Bolivia_129334",
        "type": "Sentinel-1",
        "url": "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FS1Hand%2FBolivia_129334_S1Hand.tif?generation=1601572685647701&alt=media",
        "dest": Path("dataset/raw/Bolivia_129334_S1Hand.tif"),
        "expected_bytes": 1764952,
    },
    {
        "chip": "Bolivia_129334",
        "type": "Sentinel-2",
        "url": "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FS2Hand%2FBolivia_129334_S2Hand.tif?generation=1601573238158344&alt=media",
        "dest": Path("dataset/raw/Bolivia_129334_S2Hand.tif"),
        "expected_bytes": 2191905,
    },
    {
        "chip": "Bolivia_129334",
        "type": "LabelHand",
        "url": "https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/v1.1%2Fdata%2Fflood_events%2FHandLabeled%2FLabelHand%2FBolivia_129334_LabelHand.tif?generation=1601572621201837&alt=media",
        "dest": Path("dataset/labels/Bolivia_129334_LabelHand.tif"),
        "expected_bytes": 13085,
    },
]

def run_download():
    print("=" * 70)
    print("Downloading 2-Chip Verified Sample Set from Sen1Floods11 (v1.1)")
    print("=" * 70)
    
    success_count = 0
    for item in TARGETS:
        dest_path = item["dest"]
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading [{item['type']}] -> {dest_path} ... ", end="", flush=True)

        try:
            urllib.request.urlretrieve(item["url"], dest_path)
            actual_size = dest_path.stat().st_size
            if actual_size == item["expected_bytes"]:
                print(f"SUCCESS ({actual_size} bytes, exact match)")
                success_count += 1
            else:
                print(f"WARNING: Size mismatch (got {actual_size}, expected {item['expected_bytes']})")
                success_count += 1
        except Exception as e:
            print(f"FAILED: {e}")
            if dest_path.exists():
                dest_path.unlink()

    print("=" * 70)
    print(f"Download complete: {success_count}/{len(TARGETS)} files acquired.")

if __name__ == "__main__":
    run_download()
