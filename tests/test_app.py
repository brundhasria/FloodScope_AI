"""
FloodScope AI - Unit & Integration Tests for Web Backend & Detection API
Member 3: Application & Integration

Tests all API endpoints, input validations, error states, and security guards:
1. Health and environment status endpoint.
2. Sample datasets metadata endpoint.
3. Synchronous detection on verified sample tile (default baseline and custom parameters).
4. Validation and error handling for invalid thresholds, unknown methods, and non-existent datasets.
5. Upload-based detection on synthetic image pairs.
6. Rejection of dimension mismatches, corrupt files, and 0-byte files.
7. Proper exclusion of invalid pixels using uploaded overlap masks.
8. Artifact serving with security guards against path traversal.
9. Mandatory preservation of scientific disclaimer warnings in API responses.
10. Frontend dashboard HTML delivery.
"""

import io
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from starlette.testclient import TestClient

# Ensure src/ is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from floodscope.app import app


class TestApplicationAPI(unittest.TestCase):
    """Test suite for FloodScope AI FastAPI application."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # ---------------------------------------------------------------------
    # Health and Metadata Endpoints
    # ---------------------------------------------------------------------

    def test_1_health_endpoint(self):
        """Verify GET /api/health returns 200, valid versions, and status 'healthy'."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("version", data)
        self.assertIn("opencv_version", data)
        self.assertIn("python_version", data)
        self.assertTrue(data["sample_data_available"])

    def test_2_sample_datasets_endpoint(self):
        """Verify GET /api/sample-datasets lists the verified SpaceNet 8 tile."""
        res = self.client.get("/api/sample-datasets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 1)
        sample = data["samples"][0]
        self.assertEqual(sample["id"], "SpaceNet8_Germany_0_16_67")
        self.assertIn("artifacts", sample)
        self.assertIn("pre_image_url", sample["artifacts"])
        self.assertIn("post_image_url", sample["artifacts"])

    # ---------------------------------------------------------------------
    # Sample Detection Endpoint
    # ---------------------------------------------------------------------

    def test_3_detect_sample_default_baseline(self):
        """Verify POST /api/detect/sample with default parameters returns pre-computed baseline."""
        payload = {
            "sample_id": "SpaceNet8_Germany_0_16_67",
            "threshold": 35.0,
            "method": "grayscale_diff",
            "apply_morphology": True,
            "morph_kernel_size": 3,
        }
        res = self.client.post("/api/detect/sample", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["run_id"], "baseline")
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["valid_pixels"], 1690000)
        self.assertEqual(data["summary"]["changed_pixels"], 477042)
        self.assertAlmostEqual(data["summary"]["valid_area_change_pct"], 28.2273, places=3)
        self.assertIn("warnings", data)
        self.assertTrue(any("Candidate changes" in w or "candidate changes" in w for w in data["warnings"]))
        self.assertIn("artifacts", data)
        self.assertIn("overlay_url", data["artifacts"])

    def test_4_detect_sample_custom_threshold(self):
        """Verify POST /api/detect/sample with custom threshold executes dynamically."""
        payload = {
            "sample_id": "SpaceNet8_Germany_0_16_67",
            "threshold": 60.0,
            "method": "grayscale_diff",
            "apply_morphology": True,
            "morph_kernel_size": 3,
        }
        res = self.client.post("/api/detect/sample", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(data["run_id"].startswith("sample_"))
        # Higher threshold must produce fewer changed pixels than threshold=35
        self.assertLess(data["summary"]["changed_pixels"], 477042)
        self.assertIn("artifacts", data)
        self.assertTrue(data["artifacts"]["change_mask_url"].endswith(".png"))

    def test_5_detect_sample_color_diff_method(self):
        """Verify POST /api/detect/sample with 'color_diff' runs Euclidean distance formulation."""
        payload = {
            "sample_id": "SpaceNet8_Germany_0_16_67",
            "threshold": 35.0,
            "method": "color_diff",
            "apply_morphology": False,
            "morph_kernel_size": 3,
        }
        res = self.client.post("/api/detect/sample", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["summary"]["method"], "color_diff")

    def test_6_detect_sample_invalid_threshold_rejected(self):
        """Negative threshold must be rejected with 422 or 400 error."""
        payload = {
            "sample_id": "SpaceNet8_Germany_0_16_67",
            "threshold": -10.0,
        }
        res = self.client.post("/api/detect/sample", json=payload)
        self.assertIn(res.status_code, [400, 422])

    def test_7_detect_sample_invalid_method_rejected(self):
        """Unknown difference method must be rejected with 400 error."""
        payload = {
            "sample_id": "SpaceNet8_Germany_0_16_67",
            "threshold": 35.0,
            "method": "magic_ai_diff",
        }
        res = self.client.post("/api/detect/sample", json=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid detection configuration", res.json()["detail"])

    def test_8_detect_sample_nonexistent_dataset_rejected(self):
        """Request for unknown sample_id must return 404."""
        payload = {
            "sample_id": "unknown_tile_12345",
        }
        res = self.client.post("/api/detect/sample", json=payload)
        self.assertEqual(res.status_code, 404)

    # ---------------------------------------------------------------------
    # Upload Detection Endpoint
    # ---------------------------------------------------------------------

    def _encode_png(self, arr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", arr)
        self.assertTrue(ok)
        return buf.tobytes()

    def test_9_detect_upload_valid_synthetic_pair(self):
        """Verify upload of valid matching synthetic images produces successful detection."""
        base_img = np.full((120, 120, 3), 100, dtype=np.uint8)
        changed_img = base_img.copy()
        # Add 30x30 change (900 pixels)
        changed_img[20:50, 20:50] = 200

        bytes_pre = self._encode_png(base_img)
        bytes_post = self._encode_png(changed_img)

        files = {
            "before_image": ("pre.png", bytes_pre, "image/png"),
            "after_image": ("post.png", bytes_post, "image/png"),
        }
        data = {
            "threshold": "35.0",
            "method": "grayscale_diff",
            "apply_morphology": "false",
            "morph_kernel_size": "3",
        }

        res = self.client.post("/api/detect/upload", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        result = res.json()
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["summary"]["changed_pixels"], 900)
        self.assertEqual(result["summary"]["total_pixels"], 120 * 120)
        self.assertTrue(result["run_id"].startswith("upload_"))

    def test_10_detect_upload_dimension_mismatch_rejected(self):
        """Mismatched before and after image dimensions must be rejected with 400."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.zeros((120, 120, 3), dtype=np.uint8)

        files = {
            "before_image": ("pre.png", self._encode_png(img1), "image/png"),
            "after_image": ("post.png", self._encode_png(img2), "image/png"),
        }
        res = self.client.post("/api/detect/upload", files=files, data={"threshold": "35.0"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("dimensions mismatch", res.json()["detail"].lower())

    def test_11_detect_upload_corrupt_file_rejected(self):
        """Corrupt or non-image files must be rejected gracefully with 400."""
        valid_img = np.zeros((50, 50, 3), dtype=np.uint8)
        corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_CORRUPT"

        files = {
            "before_image": ("pre.png", self._encode_png(valid_img), "image/png"),
            "after_image": ("corrupt.png", corrupt_bytes, "image/png"),
        }
        res = self.client.post("/api/detect/upload", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("failed to decode", res.json()["detail"].lower())

    def test_12_detect_upload_empty_file_rejected(self):
        """0-byte uploaded files must be rejected with 400."""
        valid_img = np.zeros((50, 50, 3), dtype=np.uint8)
        empty_bytes = b""

        files = {
            "before_image": ("pre.png", empty_bytes, "image/png"),
            "after_image": ("post.png", self._encode_png(valid_img), "image/png"),
        }
        res = self.client.post("/api/detect/upload", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("0 bytes", res.json()["detail"].lower())

    def test_13_detect_upload_with_overlap_mask_excludes_invalid_regions(self):
        """Invalid mask pixels must be strictly excluded from detection metrics."""
        base_img = np.full((100, 100, 3), 100, dtype=np.uint8)
        changed_img = base_img.copy()
        # Change in top-left 25x25 region
        changed_img[:25, :25] = 220

        # Mask top-left 25x25 as INVALID (0)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        mask[:25, :25] = 0

        files = {
            "before_image": ("pre.png", self._encode_png(base_img), "image/png"),
            "after_image": ("post.png", self._encode_png(changed_img), "image/png"),
            "overlap_mask": ("mask.png", self._encode_png(mask), "image/png"),
        }
        res = self.client.post("/api/detect/upload", files=files, data={"threshold": "20.0"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["summary"]["changed_pixels"], 0)
        self.assertEqual(data["summary"]["valid_area_change_pct"], 0.0)
        self.assertEqual(data["summary"]["invalid_pixels"], 25 * 25)

    def test_14_detect_upload_mismatched_mask_dimensions_rejected(self):
        """Mask with dimensions different from images must be rejected with 400."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask_diff = np.ones((80, 80), dtype=np.uint8) * 255

        files = {
            "before_image": ("pre.png", self._encode_png(img), "image/png"),
            "after_image": ("post.png", self._encode_png(img), "image/png"),
            "overlap_mask": ("mask.png", self._encode_png(mask_diff), "image/png"),
        }
        res = self.client.post("/api/detect/upload", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("overlap mask dimensions", res.json()["detail"].lower())

    # ---------------------------------------------------------------------
    # Artifact Serving & Security
    # ---------------------------------------------------------------------

    def test_15_output_artifact_serving_baseline(self):
        """Verify GET /api/outputs/baseline/{filename} safely serves existing baseline images."""
        res = self.client.get("/api/outputs/baseline/SpaceNet8_Germany_0_16_67_candidate_change_mask.png")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["content-type"], "image/png")
        self.assertGreater(len(res.content), 0)

    def test_16_output_artifact_path_traversal_blocked(self):
        """Path traversal patterns in filename or run_id must be blocked."""
        # Dot-dot traversal in filename
        res1 = self.client.get("/api/outputs/baseline/..%2F..%2FREADME.md")
        self.assertIn(res1.status_code, [400, 403, 404])

        # Dot-dot in run_id
        res2 = self.client.get("/api/outputs/..%2F..%2Fetc/target.png")
        self.assertIn(res2.status_code, [400, 403, 404])

    def test_17_scientific_warnings_always_present_in_api(self):
        """Every successful detection response must contain explicit scientific interpretation warnings."""
        res = self.client.post("/api/detect/sample", json={"sample_id": "SpaceNet8_Germany_0_16_67"})
        self.assertEqual(res.status_code, 200)
        warnings = res.json().get("warnings", [])
        self.assertGreater(len(warnings), 0)
        text = " ".join(warnings)
        self.assertIn("NOT confirmed floodwater", text)

    def test_18_index_page_delivered(self):
        """Verify GET / delivers the main HTML dashboard with required scientific notices."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.text
        self.assertIn("FloodScope AI", html)
        self.assertIn("Candidate Surface Changes — Not Confirmed Flooding", html)
        self.assertIn("app.js", html)
        self.assertIn("style.css", html)


if __name__ == "__main__":
    unittest.main()
