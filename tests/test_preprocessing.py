"""
FloodScope AI - Unit and Integration Tests for Image Preprocessing and Alignment
Member 2: Detection

Runs via standard library: `python -m unittest tests/test_preprocessing.py`
Also compatible with `pytest`.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from floodscope.preprocessing import (
    SpatialAlignmentError,
    align_image_pair,
    compute_geospatial_affine_matrix,
    evaluate_spatial_compatibility,
    extract_geotiff_metadata,
    generate_alignment_diagnostics,
    load_image,
    save_alignment_report,
)

RAW_DIR = repo_root / "dataset" / "raw"
PRE_FILE = RAW_DIR / "SpaceNet8_Germany_0_16_67_PRE.tif"
POST_FILE = RAW_DIR / "SpaceNet8_Germany_0_16_67_POST.tif"
S1_FILE = RAW_DIR / "Bolivia_103757_S1Hand.tif"
S2_FILE = RAW_DIR / "Bolivia_103757_S2Hand.tif"


class TestImageLoading(unittest.TestCase):
    """Test loading RGB, multi-band optical, SAR, missing and corrupt images."""

    def test_load_image_rgb_optical(self):
        self.assertTrue(PRE_FILE.is_file(), f"Missing file {PRE_FILE}")
        img, meta = load_image(PRE_FILE)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (1300, 1300, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(meta["channels"], 3)
        self.assertEqual(meta["height"], 1300)
        self.assertEqual(meta["width"], 1300)

    def test_load_image_multiband_optical(self):
        self.assertTrue(S2_FILE.is_file(), f"Missing file {S2_FILE}")
        img, meta = load_image(S2_FILE)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (512, 512, 13))
        self.assertEqual(meta["channels"], 13)
        self.assertIn(meta["dtype"], ["int16", "int32"])

    def test_load_image_multipolarization_sar(self):
        self.assertTrue(S1_FILE.is_file(), f"Missing file {S1_FILE}")
        img, meta = load_image(S1_FILE)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (512, 512, 2))
        self.assertEqual(meta["channels"], 2)
        self.assertEqual(img.dtype, np.float32)

    def test_load_image_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_image(repo_root / "dataset" / "raw" / "non_existent_file.tif")

    def test_load_image_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError):
                load_image(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestGeotiffMetadataAndCompatibility(unittest.TestCase):
    """Test GeoTIFF metadata extraction and geospatial compatibility checks."""

    def test_extract_geotiff_metadata_real(self):
        meta = extract_geotiff_metadata(PRE_FILE)
        self.assertTrue(meta["has_georeference"])
        self.assertEqual(meta["crs"], "EPSG:4326")
        self.assertEqual(meta["crs_epsg"], 4326)
        self.assertEqual(len(meta["tiepoint"]), 6)
        self.assertEqual(len(meta["pixel_scale"]), 3)
        self.assertTrue(6.0 < meta["bounds"]["lon_min"] < 7.5)
        self.assertTrue(50.0 < meta["bounds"]["lat_min"] < 51.0)
        self.assertEqual(meta["width"], 1300)
        self.assertEqual(meta["height"], 1300)

    def test_spatial_compatibility_real_spacenet8(self):
        meta_pre = extract_geotiff_metadata(PRE_FILE)
        meta_post = extract_geotiff_metadata(POST_FILE)
        comp = evaluate_spatial_compatibility(meta_pre, meta_post)
        self.assertTrue(comp["is_compatible"])
        self.assertTrue(comp["crs_matched"])
        self.assertGreater(comp["overlap_pct_pre"], 99.0)
        self.assertGreater(comp["overlap_pct_post"], 99.0)
        self.assertIsNotNone(comp["intersection_bounds"])

    def test_synthetic_spatial_compatibility_mismatched_crs(self):
        meta_pre = {
            "has_georeference": True,
            "crs": "EPSG:4326",
            "bounds": {"lon_min": 10.0, "lon_max": 11.0, "lat_min": 50.0, "lat_max": 51.0},
            "pixel_scale": [0.001, 0.001, 0.0],
            "tiepoint": [0.0, 0.0, 0.0, 10.0, 51.0, 0.0],
        }
        meta_post = {
            "has_georeference": True,
            "crs": "EPSG:32632",  # Mismatched UTM zone
            "bounds": {"lon_min": 10.0, "lon_max": 11.0, "lat_min": 50.0, "lat_max": 51.0},
            "pixel_scale": [0.001, 0.001, 0.0],
            "tiepoint": [0.0, 0.0, 0.0, 10.0, 51.0, 0.0],
        }
        comp = evaluate_spatial_compatibility(meta_pre, meta_post)
        self.assertFalse(comp["is_compatible"])
        self.assertFalse(comp["crs_matched"])
        self.assertTrue(any("CRS mismatch" in w for w in comp["warnings"]))

    def test_synthetic_spatial_compatibility_disjoint_scenes(self):
        meta_pre = {
            "has_georeference": True,
            "crs": "EPSG:4326",
            "bounds": {"lon_min": 10.0, "lon_max": 11.0, "lat_min": 50.0, "lat_max": 51.0},
            "pixel_scale": [0.001, 0.001, 0.0],
            "tiepoint": [0.0, 0.0, 0.0, 10.0, 51.0, 0.0],
        }
        meta_post = {
            "has_georeference": True,
            "crs": "EPSG:4326",
            "bounds": {"lon_min": 20.0, "lon_max": 21.0, "lat_min": 60.0, "lat_max": 61.0},  # Disjoint
            "pixel_scale": [0.001, 0.001, 0.0],
            "tiepoint": [0.0, 0.0, 0.0, 20.0, 61.0, 0.0],
        }
        comp = evaluate_spatial_compatibility(meta_pre, meta_post)
        self.assertFalse(comp["is_compatible"])
        self.assertTrue(any("zero geographic overlap" in w for w in comp["warnings"]))


class TestAffineTransformAndAlignment(unittest.TestCase):
    """Test affine transform computation and end-to-end alignment."""

    def test_compute_affine_matrix(self):
        meta_pre = extract_geotiff_metadata(PRE_FILE)
        meta_post = extract_geotiff_metadata(POST_FILE)
        # Forward matrix maps POST coordinates (1743x1743) -> PRE coordinates (1300x1300)
        M = compute_geospatial_affine_matrix(meta_src=meta_post, meta_dst=meta_pre)
        self.assertEqual(M.shape, (2, 3))
        # The forward scale factor is scale_post / scale_pre ~ 0.7464
        self.assertTrue(0.74 < M[0, 0] < 0.75)
        self.assertTrue(0.74 < M[1, 1] < 0.75)
        self.assertEqual(M[0, 1], 0.0)
        self.assertEqual(M[1, 0], 0.0)
        # Translation offset in PRE pixels is sub-pixel (< 1.0 px)
        self.assertTrue(abs(M[0, 2]) < 1.0)
        self.assertTrue(abs(M[1, 2]) < 1.0)

    def test_synthetic_warp_affine_direction(self):
        """Verify that cv2.warpAffine with M maps known synthetic coordinates correctly."""
        src = np.zeros((10, 10), dtype=np.uint8)
        src[5, 5] = 255  # Point at (x_src=5, y_src=5)
        meta_src = {
            "tiepoint": [0, 0, 0, 0.0, 0.0, 0],
            "pixel_scale": [1.0, 1.0, 0]
        }
        meta_dst = {
            "tiepoint": [0, 0, 0, 2.0, 2.0, 0],
            "pixel_scale": [1.0, 1.0, 0]
        }
        M = compute_geospatial_affine_matrix(meta_src=meta_src, meta_dst=meta_dst)
        # Expected dst coord: lon = 0 + 5*1 = 5 => x_dst = (5-2)/1 = 3
        # lat = 0 - 5*1 = -5 => y_dst = (2 - (-5))/1 = 7
        dst = cv2.warpAffine(src, M, (10, 10), flags=cv2.INTER_NEAREST)
        y_idx, x_idx = np.where(dst == 255)
        self.assertEqual(len(x_idx), 1)
        self.assertEqual(x_idx[0], 3)
        self.assertEqual(y_idx[0], 7)

    def test_align_image_pair_real_data(self):
        result = align_image_pair(PRE_FILE, POST_FILE, target_grid="pre")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["pre_aligned"].shape, (1300, 1300, 3))
        self.assertEqual(result["post_aligned"].shape, (1300, 1300, 3))
        self.assertEqual(result["overlap_mask"].shape, (1300, 1300))
        self.assertEqual(result["overlap_mask"].dtype, np.uint8)
        self.assertGreater(result["metadata"]["valid_overlap_ratio"], 0.998)

    def test_align_image_pair_incompatible_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            p1 = Path(tmp_dir) / "img1.png"
            p2 = Path(tmp_dir) / "img2.png"
            cv2.imwrite(str(p1), np.zeros((100, 100, 3), dtype=np.uint8))
            cv2.imwrite(str(p2), np.zeros((100, 100, 3), dtype=np.uint8))
            with self.assertRaises(SpatialAlignmentError):
                align_image_pair(p1, p2)


class TestDiagnosticsAndReports(unittest.TestCase):
    """Test generating visual diagnostics and audit reports."""

    def test_generate_alignment_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pre_fake = np.full((200, 200, 3), 100, dtype=np.uint8)
            post_fake = np.full((200, 200, 3), 200, dtype=np.uint8)

            paths = generate_alignment_diagnostics(
                pre_aligned=pre_fake,
                post_aligned=post_fake,
                output_dir=tmp_dir,
                prefix="test_synth",
                square_size=40,
            )

            self.assertTrue(paths["side_by_side"].is_file())
            self.assertTrue(paths["overlay"].is_file())
            self.assertTrue(paths["checkerboard"].is_file())
            self.assertTrue(paths["absdiff"].is_file())

            sbs = cv2.imread(str(paths["side_by_side"]))
            self.assertEqual(sbs.shape, (200, 400, 3))

            chk = cv2.imread(str(paths["checkerboard"]))
            self.assertEqual(chk[10, 10, 0], 100)
            self.assertEqual(chk[10, 50, 0], 200)

    def test_save_alignment_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rep_path = Path(tmp_dir) / "report.json"
            meta = {"status": "SUCCESS", "ratio": 0.999}
            saved = save_alignment_report(meta, rep_path)
            self.assertTrue(saved.is_file())
            with open(saved, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "SUCCESS")
            self.assertEqual(data["ratio"], 0.999)


if __name__ == "__main__":
    unittest.main()
