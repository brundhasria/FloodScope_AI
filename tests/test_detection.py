"""
FloodScope AI - Unit and Integration Tests for Change Detection Baseline
Member 2: Detection

Tests:
1. Identical images produce zero candidate changes.
2. Synthetic image with a known changed rectangle produces detections in the exact region.
3. Artificial differences in invalid mask regions are strictly excluded.
4. Valid changes inside the overlap region are counted.
5. Missing input files produce status='FAILED' with clear error messages.
6. Empty or unreadable files produce status='FAILED' with clear error messages.
7. Mismatched dimensions are rejected without silent resizing.
8. Mismatched mask dimensions are rejected.
9. Zero valid pixels in the mask are handled safely (no ZeroDivisionError).
10. Returned mask and overlay have expected dimensions, shapes, and types.
11. Summary counts and percentages are computed strictly over valid pixels.
12. DetectionConfig validates parameters (rejects negative thresholds, invalid methods).
13. Output artifact writing works in a temporary directory.
14. The public detect_changes() function returns the standardized DetectionResult contract.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(repo_root / "src"))

from floodscope.detection import (
    DetectionConfig,
    DetectionResult,
    detect_changes,
    save_detection_artifacts,
)

PROC_DIR = repo_root / "dataset" / "processed"
PRE_ALIGNED = PROC_DIR / "SpaceNet8_Germany_0_16_67_PRE_aligned.png"
POST_ALIGNED = PROC_DIR / "SpaceNet8_Germany_0_16_67_POST_aligned.png"
OVERLAP_MASK = PROC_DIR / "SpaceNet8_Germany_0_16_67_overlap_mask.png"


class TestDetectionBaseline(unittest.TestCase):
    """Test suite for change detection baseline."""

    def setUp(self):
        # Create standard synthetic test canvas (100x100 RGB)
        self.base_img = np.full((100, 100, 3), 120, dtype=np.uint8)
        self.valid_mask = np.ones((100, 100), dtype=np.uint8) * 255

    def test_1_identical_images_produce_zero_changes(self):
        """Identical PRE and POST images must yield 0 changed pixels."""
        res = detect_changes(
            before_path=self.base_img,
            after_path=self.base_img,
            overlap_mask_path=self.valid_mask,
            config=DetectionConfig(threshold=10.0),
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.summary["changed_pixels"], 0)
        self.assertEqual(res.summary["valid_area_change_pct"], 0.0)
        self.assertEqual(np.count_nonzero(res.change_mask), 0)

    def test_2_synthetic_known_rectangle_change(self):
        """A known changed rectangle must be detected in the exact region."""
        changed_img = self.base_img.copy()
        # Add a 30x30 bright square at [20:50, 40:70]
        changed_img[20:50, 40:70] = 220  # Delta = 100
        res = detect_changes(
            before_path=self.base_img,
            after_path=changed_img,
            overlap_mask_path=self.valid_mask,
            config=DetectionConfig(threshold=35.0, apply_morphology=False),
        )
        self.assertEqual(res.status, "SUCCESS")
        expected_changed_pixels = 30 * 30  # 900
        self.assertEqual(res.summary["changed_pixels"], expected_changed_pixels)
        # Verify detected pixels match the rectangle
        detected_coords = np.where(res.change_mask == 255)
        self.assertEqual(detected_coords[0].min(), 20)
        self.assertEqual(detected_coords[0].max(), 49)
        self.assertEqual(detected_coords[1].min(), 40)
        self.assertEqual(detected_coords[1].max(), 69)

    def test_3_difference_only_in_invalid_pixels_excluded(self):
        """Changes occurring solely in invalid mask regions must NOT be detected."""
        changed_img = self.base_img.copy()
        # Change the top-left 20x20 pixels
        changed_img[:20, :20] = 255
        # Set mask to INVALID (0) over that exact 20x20 region
        partial_mask = self.valid_mask.copy()
        partial_mask[:20, :20] = 0

        res = detect_changes(
            before_path=self.base_img,
            after_path=changed_img,
            overlap_mask_path=partial_mask,
            config=DetectionConfig(threshold=20.0),
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.summary["changed_pixels"], 0)
        self.assertEqual(res.summary["valid_area_change_pct"], 0.0)
        self.assertEqual(np.count_nonzero(res.change_mask), 0)

    def test_4_valid_change_inside_overlap_counted(self):
        """Changes occurring inside valid mask regions must be detected."""
        changed_img = self.base_img.copy()
        changed_img[60:80, 60:80] = 250  # 20x20 = 400 pixels
        res = detect_changes(
            before_path=self.base_img,
            after_path=changed_img,
            overlap_mask_path=self.valid_mask,
            config=DetectionConfig(threshold=30.0, apply_morphology=False),
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.summary["changed_pixels"], 400)
        self.assertEqual(res.summary["valid_area_change_pct"], 4.0)

    def test_5_missing_image_files_failure(self):
        """Non-existent image paths must return status='FAILED' with clear error."""
        res = detect_changes(
            before_path=Path("non_existent_before.png"),
            after_path=self.base_img,
        )
        self.assertEqual(res.status, "FAILED")
        self.assertIn("does not exist", res.error)

    def test_6_corrupt_or_unreadable_inputs_failure(self):
        """0-byte image files must return status='FAILED'."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            res = detect_changes(before_path=tmp_path, after_path=self.base_img)
            self.assertEqual(res.status, "FAILED")
            self.assertIn("0 bytes", res.error)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_7_mismatched_dimensions_rejected(self):
        """Different image dimensions must be rejected rather than resized."""
        img_different = np.zeros((150, 150, 3), dtype=np.uint8)
        res = detect_changes(before_path=self.base_img, after_path=img_different)
        self.assertEqual(res.status, "FAILED")
        self.assertIn("dimensions mismatch", res.error)

    def test_8_invalid_mask_dimensions_handled(self):
        """A mask with dimensions mismatched to the images must fail gracefully."""
        mask_different = np.ones((80, 80), dtype=np.uint8) * 255
        res = detect_changes(
            before_path=self.base_img,
            after_path=self.base_img,
            overlap_mask_path=mask_different,
        )
        self.assertEqual(res.status, "FAILED")
        self.assertIn("mask dimensions", res.error)

    def test_9_zero_valid_pixels_handled_safely(self):
        """An all-zero mask must not cause ZeroDivisionError and must report 0.0% change."""
        zero_mask = np.zeros((100, 100), dtype=np.uint8)
        changed_img = self.base_img.copy()
        changed_img[10:30, 10:30] = 250
        res = detect_changes(
            before_path=self.base_img,
            after_path=changed_img,
            overlap_mask_path=zero_mask,
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.summary["valid_pixels"], 0)
        self.assertEqual(res.summary["changed_pixels"], 0)
        self.assertEqual(res.summary["valid_area_change_pct"], 0.0)

    def test_10_returned_mask_and_overlay_dimensions_and_types(self):
        """Verify return array shapes, dimensions, and data types."""
        res = detect_changes(
            before_path=self.base_img,
            after_path=self.base_img,
            overlap_mask_path=self.valid_mask,
        )
        self.assertEqual(res.change_mask.shape, (100, 100))
        self.assertEqual(res.change_mask.dtype, np.uint8)
        self.assertEqual(res.overlay.shape, (100, 100, 3))
        self.assertEqual(res.overlay.dtype, np.uint8)
        self.assertEqual(res.difference_image.shape, (100, 100))
        self.assertEqual(res.difference_image.dtype, np.uint8)

    def test_11_summary_metrics_calculated_on_valid_pixels_only(self):
        """Check summary counts and math consistency."""
        half_mask = np.zeros((100, 100), dtype=np.uint8)
        half_mask[:50, :] = 255  # 5,000 valid pixels
        changed = self.base_img.copy()
        changed[:25, :] = 255  # 2,500 changed pixels in valid region
        changed[75:, :] = 255  # Changed in invalid region (must be ignored)

        res = detect_changes(
            before_path=self.base_img,
            after_path=changed,
            overlap_mask_path=half_mask,
            config=DetectionConfig(threshold=20.0, apply_morphology=False),
        )
        self.assertEqual(res.summary["total_pixels"], 10000)
        self.assertEqual(res.summary["valid_pixels"], 5000)
        self.assertEqual(res.summary["invalid_pixels"], 5000)
        self.assertEqual(res.summary["changed_pixels"], 2500)
        self.assertAlmostEqual(res.summary["valid_area_change_pct"], 50.0, places=2)

    def test_12_configuration_values_validated(self):
        """Invalid configurations must be rejected."""
        res_neg = detect_changes(
            before_path=self.base_img,
            after_path=self.base_img,
            config=DetectionConfig(threshold=-5.0),
        )
        self.assertEqual(res_neg.status, "FAILED")
        self.assertIn("Invalid DetectionConfig", res_neg.error)

        res_meth = detect_changes(
            before_path=self.base_img,
            after_path=self.base_img,
            config=DetectionConfig(method="invalid_method"),
        )
        self.assertEqual(res_meth.status, "FAILED")

    def test_13_save_artifacts_in_temporary_directory(self):
        """Verify artifact saving produces expected files and valid JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = detect_changes(
                before_path=self.base_img,
                after_path=self.base_img,
                overlap_mask_path=self.valid_mask,
            )
            paths = save_detection_artifacts(res, tmp_dir, prefix="test_case")
            for k, p in paths.items():
                self.assertTrue(p.is_file(), f"Missing output file for {k}")

            with open(paths["summary"], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "SUCCESS")
            self.assertIn("summary", data)
            self.assertIn("warnings", data)

    def test_14_public_handoff_function_contract(self):
        """Verify detect_changes() returns the complete DetectionResult dataclass."""
        res = detect_changes(
            before_path=self.base_img,
            after_path=self.base_img,
            overlap_mask_path=self.valid_mask,
        )
        self.assertIsInstance(res, DetectionResult)
        self.assertIn(res.status, ["SUCCESS", "FAILED"])
        self.assertIsInstance(res.change_mask, np.ndarray)
        self.assertIsInstance(res.overlay, np.ndarray)
        self.assertIsInstance(res.difference_image, np.ndarray)
        self.assertIsInstance(res.summary, dict)
        self.assertIsInstance(res.warnings, list)
        self.assertIsInstance(res.metadata, dict)
        self.assertGreater(len(res.warnings), 0)
        self.assertTrue(any("candidate changes" in w for w in res.warnings))


class TestRealDataIntegration(unittest.TestCase):
    """Integration test against actual processed SpaceNet 8 files."""

    def test_real_spacenet8_detection(self):
        if not PRE_ALIGNED.is_file() or not POST_ALIGNED.is_file() or not OVERLAP_MASK.is_file():
            self.skipTest("Aligned SpaceNet 8 files not yet present in dataset/processed/")

        res = detect_changes(
            before_path=PRE_ALIGNED,
            after_path=POST_ALIGNED,
            overlap_mask_path=OVERLAP_MASK,
            config=DetectionConfig(threshold=35.0),
        )
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.change_mask.shape, (1300, 1300))
        self.assertEqual(res.summary["valid_pixels"], 1690000)
        self.assertGreater(res.summary["changed_pixels"], 0)
        self.assertLess(res.summary["changed_pixels"], 1690000)


if __name__ == "__main__":
    unittest.main()
