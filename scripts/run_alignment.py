"""
FloodScope AI - Run Image Preprocessing and Spatial Alignment
Member 2: Detection

Executes the geospatial alignment pipeline on the SpaceNet 8 bi-temporal pair,
saves the aligned images, generates diagnostic visual checks, and outputs
a machine-readable alignment audit report.
"""

import sys
from pathlib import Path

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from floodscope.preprocessing import (
    align_image_pair,
    generate_alignment_diagnostics,
    save_alignment_report,
)
import cv2


def main():
    print("=" * 70)
    print("FloodScope AI -- Stage 2 Geospatial Alignment Pipeline")
    print("=" * 70)

    pre_path = repo_root / "dataset" / "raw" / "SpaceNet8_Germany_0_16_67_PRE.tif"
    post_path = repo_root / "dataset" / "raw" / "SpaceNet8_Germany_0_16_67_POST.tif"
    out_dir = repo_root / "dataset" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PRE image:  {pre_path}")
    print(f"POST image: {post_path}")
    print(f"Target dir: {out_dir}")
    print("-" * 70)

    # Run alignment onto PRE grid
    print("Executing geospatial alignment (target_grid='pre')...")
    result = align_image_pair(pre_path, post_path, target_grid="pre")

    meta = result["metadata"]
    print(f"Alignment Status:      {result['status']}")
    print(f"Target Spatial Grid:   {meta['target_grid']} ({meta['grid_dimensions'][0]}x{meta['grid_dimensions'][1]})")
    print(f"Valid Overlap Pixels:  {meta['valid_overlap_pixels']} / {meta['grid_dimensions'][0]*meta['grid_dimensions'][1]}")
    print(f"Valid Overlap Ratio:   {meta['valid_overlap_ratio']*100:.4f}%")
    print(f"CRS Matched:           {meta['compatibility']['crs_matched']} ({meta['meta_pre']['crs']})")
    print(f"Geographic Overlap:    PRE={meta['compatibility']['overlap_pct_pre']:.2f}%, POST={meta['compatibility']['overlap_pct_post']:.2f}%")
    print(f"Affine Transform M:")
    for row in meta["affine_matrix"]:
        print(f"  [{row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}]")
    print("-" * 70)

    # Save aligned raster outputs
    p_pre_out = out_dir / "SpaceNet8_Germany_0_16_67_PRE_aligned.png"
    p_post_out = out_dir / "SpaceNet8_Germany_0_16_67_POST_aligned.png"
    p_mask_out = out_dir / "SpaceNet8_Germany_0_16_67_overlap_mask.png"

    cv2.imwrite(str(p_pre_out), result["pre_aligned"])
    cv2.imwrite(str(p_post_out), result["post_aligned"])
    cv2.imwrite(str(p_mask_out), result["overlap_mask"])
    print(f"Saved aligned PRE:     {p_pre_out.name}")
    print(f"Saved aligned POST:    {p_post_out.name}")
    print(f"Saved overlap mask:    {p_mask_out.name}")

    # Generate diagnostic visualizations
    print("Generating visual alignment diagnostics...")
    diag_paths = generate_alignment_diagnostics(
        pre_aligned=result["pre_aligned"],
        post_aligned=result["post_aligned"],
        output_dir=out_dir,
        prefix="SpaceNet8_Germany_0_16_67",
        square_size=130,
    )
    for name, path in diag_paths.items():
        print(f"  - {name:15}: {path.name}")

    # Save machine-readable metadata report
    report_path = out_dir / "SpaceNet8_Germany_0_16_67_alignment_report.json"
    save_alignment_report(meta, report_path)
    print(f"Saved audit report:    {report_path.name}")

    print("=" * 70)
    print("Stage 2 Alignment Pipeline Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
