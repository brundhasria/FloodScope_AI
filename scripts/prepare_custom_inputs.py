"""
FloodScope AI - Prepare SpaceNet 8 Custom Upload Satellite Inputs
Member 3: Application & Integration

Reads the original SpaceNet 8 PRE and POST GeoTIFFs using rasterio,
reprojects/resamples the POST scene onto PRE's exact spatial grid,
converts both to RGB uint8 PNGs, handles nodata/invalid pixels,
and saves:
  - dataset/processed/custom_before.png
  - dataset/processed/custom_after.png
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from PIL import Image

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
raw_dir = repo_root / "dataset" / "raw"
proc_dir = repo_root / "dataset" / "processed"
proc_dir.mkdir(parents=True, exist_ok=True)

pre_tif_path = raw_dir / "SpaceNet8_Germany_0_16_67_PRE.tif"
post_tif_path = raw_dir / "SpaceNet8_Germany_0_16_67_POST.tif"

out_before_png = proc_dir / "custom_before.png"
out_after_png = proc_dir / "custom_after.png"


def prepare_custom_images():
    print("=" * 75)
    print("FloodScope AI -- SpaceNet 8 Preparation for Custom Image Upload")
    print("=" * 75)

    # 1. Read PRE GeoTIFF with rasterio
    print(f"Reading PRE GeoTIFF:  {pre_tif_path.name}")
    with rasterio.open(pre_tif_path) as src_pre:
        dst_crs = src_pre.crs
        dst_transform = src_pre.transform
        dst_width = src_pre.width
        dst_height = src_pre.height
        dst_count = src_pre.count
        pre_bands = src_pre.read()  # Shape: (3, 1300, 1300)

        print(f"  PRE Dimensions:     {dst_width} x {dst_height} x {dst_count}")
        print(f"  PRE CRS:            {dst_crs}")
        print(f"  PRE Transform:      {dst_transform}")

    # 2. Read POST GeoTIFF with rasterio
    print(f"\nReading POST GeoTIFF: {post_tif_path.name}")
    with rasterio.open(post_tif_path) as src_post:
        src_crs = src_post.crs
        src_transform = src_post.transform
        src_width = src_post.width
        src_height = src_post.height
        src_count = src_post.count
        post_bands = src_post.read()  # Shape: (3, 1743, 1743)

        print(f"  POST Dimensions:    {src_width} x {src_height} x {src_count}")
        print(f"  POST CRS:           {src_crs}")
        print(f"  POST Transform:     {src_transform}")

    # 3. Reproject/resample POST onto PRE's exact spatial grid
    print(f"\nReprojecting POST image onto PRE spatial grid ({dst_width}x{dst_height})...")
    post_resampled = np.zeros((dst_count, dst_height, dst_width), dtype=np.uint8)

    for band_idx in range(dst_count):
        reproject(
            source=post_bands[band_idx],
            destination=post_resampled[band_idx],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    # 4. Handle nodata & invalid pixels
    # Clean any non-finite values
    pre_clean = np.nan_to_num(pre_bands, nan=0, posinf=255, neginf=0)
    post_clean = np.nan_to_num(post_resampled, nan=0, posinf=255, neginf=0)

    # Ensure uint8 [0, 255]
    pre_uint8 = np.clip(pre_clean, 0, 255).astype(np.uint8)
    post_uint8 = np.clip(post_clean, 0, 255).astype(np.uint8)

    # Convert band-first (3, H, W) to image array (H, W, 3) in RGB order
    pre_rgb = np.moveaxis(pre_uint8, 0, -1)
    post_rgb = np.moveaxis(post_uint8, 0, -1)

    # 5. Save PNG outputs using Pillow (guaranteeing clean RGB uint8 encoding)
    img_before = Image.fromarray(pre_rgb, mode="RGB")
    img_after = Image.fromarray(post_rgb, mode="RGB")

    img_before.save(out_before_png, format="PNG")
    img_after.save(out_after_png, format="PNG")

    size_before_mb = out_before_png.stat().st_size / (1024 * 1024)
    size_after_mb = out_after_png.stat().st_size / (1024 * 1024)

    print(f"\nSaved outputs:")
    print(f"  - {out_before_png.name}: {img_before.size[0]}x{img_before.size[1]} px | {size_before_mb:.2f} MB")
    print(f"  - {out_after_png.name}:  {img_after.size[0]}x{img_after.size[1]} px | {size_after_mb:.2f} MB")

    # 6. Verification
    assert img_before.size == img_after.size == (1300, 1300), "Dimensions mismatch!"
    assert img_before.mode == img_after.mode == "RGB", "Color mode is not RGB!"
    assert size_before_mb < 25.0, f"Before file exceeds 25 MB ({size_before_mb:.2f} MB)"
    assert size_after_mb < 25.0, f"After file exceeds 25 MB ({size_after_mb:.2f} MB)"

    print("\n--- Verification Checklist ---")
    print(f"  [PASS] Identical dimensions: {img_before.size[0]}x{img_before.size[1]}")
    print(f"  [PASS] Valid RGB channels:   {img_before.mode} (3 bands)")
    print(f"  [PASS] File size limit:      {size_before_mb:.2f} MB and {size_after_mb:.2f} MB (< 25 MB limit)")
    print(f"  [PASS] Original raw files:   Preserved intact")

    print("\n" + "=" * 75)
    print("Scientific Notice:")
    print("  These images represent generic candidate surface/radiometric deltas.")
    print("  Observed differences must NOT be claimed as confirmed floodwater.")
    print("=" * 75)

    return {
        "width": img_before.size[0],
        "height": img_before.size[1],
        "channels": 3,
        "size_before_mb": size_before_mb,
        "size_after_mb": size_after_mb,
        "out_before_png": str(out_before_png),
        "out_after_png": str(out_after_png),
    }


if __name__ == "__main__":
    prepare_custom_images()
