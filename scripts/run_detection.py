"""
FloodScope AI - Run Change Detection Baseline
Member 2: Detection

Runs the explainable OpenCV change detection baseline on the real aligned SpaceNet 8
image pair, evaluates multiple candidate thresholds, and saves diagnostic products.
"""

import sys
from pathlib import Path

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from floodscope.detection import (
    DetectionConfig,
    detect_changes,
    save_detection_artifacts,
)


def main():
    print("=" * 75)
    print("FloodScope AI -- Stage 3 Change Detection Baseline Execution")
    print("=" * 75)

    proc_dir = repo_root / "dataset" / "processed"
    pre_aligned = proc_dir / "SpaceNet8_Germany_0_16_67_PRE_aligned.png"
    post_aligned = proc_dir / "SpaceNet8_Germany_0_16_67_POST_aligned.png"
    overlap_mask = proc_dir / "SpaceNet8_Germany_0_16_67_overlap_mask.png"

    if not pre_aligned.is_file() or not post_aligned.is_file() or not overlap_mask.is_file():
        print("ERROR: Aligned inputs missing from dataset/processed/. Run scripts/run_alignment.py first.")
        sys.exit(1)

    print(f"PRE Aligned:   {pre_aligned.name}")
    print(f"POST Aligned:  {post_aligned.name}")
    print(f"Overlap Mask:  {overlap_mask.name}")
    print("-" * 75)

    # Multi-threshold experiment: 25, 35 (default baseline), 50, 75
    thresholds = [25.0, 35.0, 50.0, 75.0]

    for t in thresholds:
        cfg = DetectionConfig(
            threshold=t,
            method="grayscale_diff",
            apply_morphology=True,
            morph_kernel_size=3,
        )
        result = detect_changes(
            before_path=pre_aligned,
            after_path=post_aligned,
            overlap_mask_path=overlap_mask,
            config=cfg,
        )

        if result.status != "SUCCESS":
            print(f"[FAILED] Detection failed for threshold={t}: {result.error}")
            continue

        s = result.summary
        print(f"Threshold = {t:4.1f} | Valid Pixels: {s['valid_pixels']} | "
              f"Changed Pixels: {s['changed_pixels']:7} | Change %: {s['valid_area_change_pct']:6.2f}%")

        prefix = f"SpaceNet8_Germany_0_16_67_t{int(t)}"
        paths = save_detection_artifacts(result, proc_dir, prefix=prefix)

    # Save default baseline artifacts with clean standard prefix
    default_cfg = DetectionConfig(threshold=35.0, method="grayscale_diff")
    default_res = detect_changes(pre_aligned, post_aligned, overlap_mask, default_cfg)
    save_detection_artifacts(default_res, proc_dir, prefix="SpaceNet8_Germany_0_16_67")

    print("-" * 75)
    print("Baseline outputs saved to dataset/processed/:")
    print("  - Mask:             SpaceNet8_Germany_0_16_67_candidate_change_mask.png")
    print("  - Overlay:          SpaceNet8_Germany_0_16_67_change_overlay.png")
    print("  - Difference Image: SpaceNet8_Germany_0_16_67_difference_image.png")
    print("  - Summary JSON:     SpaceNet8_Germany_0_16_67_detection_summary.json")
    print("=" * 75)


if __name__ == "__main__":
    main()
