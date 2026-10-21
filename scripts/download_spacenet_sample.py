"""
FloodScope AI - SpaceNet 8 Sample Downloader
Member 1: Dataset Preparation and Verification

Downloads the verified bi-temporal tile pair (0_16_67) from the official
SpaceNet 8 AWS S3 bucket over public HTTPS.
"""

import urllib.request
from pathlib import Path

FILES_TO_DOWNLOAD = [
    {
        "description": "SpaceNet 8 Germany Pre-Event GeoTIFF",
        "url": "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/Germany_Training_Public/PRE-event/10500500C4DD7000_0_16_67.tif",
        "dest": Path("dataset/raw/SpaceNet8_Germany_0_16_67_PRE.tif"),
        "expected_bytes": 3330496,
    },
    {
        "description": "SpaceNet 8 Germany Post-Event GeoTIFF",
        "url": "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/Germany_Training_Public/POST-event/1040050035DC3B00_0_16_67.tif",
        "dest": Path("dataset/raw/SpaceNet8_Germany_0_16_67_POST.tif"),
        "expected_bytes": 9124983,
    },
    {
        "description": "SpaceNet 8 Germany Annotations GeoJSON",
        "url": "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/Germany_Training_Public/annotations/0_16_67.geojson",
        "dest": Path("dataset/labels/SpaceNet8_Germany_0_16_67_annotations.geojson"),
        "expected_bytes": 8591,
    },
]

def download():
    print("=" * 70)
    print("Downloading SpaceNet 8 Germany Bi-Temporal Tile Pair (0_16_67)")
    print("=" * 70)
    
    for item in FILES_TO_DOWNLOAD:
        dest = item["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading [{item['description']}] -> {dest} ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(item["url"], dest)
            size = dest.stat().st_size
            if size == item["expected_bytes"]:
                print(f"SUCCESS ({size} bytes, exact match)")
            else:
                print(f"DONE ({size} bytes, expected {item['expected_bytes']})")
        except Exception as e:
            print(f"FAILED: {e}")
            if dest.exists():
                dest.unlink()

if __name__ == "__main__":
    download()
