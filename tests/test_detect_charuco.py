import unittest
import os
import numpy as np
from detect_charuco import CameraCalibration

class TestCharucoDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Resolve image path relative to this test script
        test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.image_path = os.path.join(test_dir, "data", "leftCameraBuffer.png")
        if not os.path.exists(cls.image_path):
            raise FileNotFoundError(f"Test image not found at '{cls.image_path}'")
            
        cls.calib = CameraCalibration(
            image_path=cls.image_path,
            squares_x=7,
            squares_y=5,
            square_length=3.5,
            marker_length=2.0,
            corner_refinement_method="apriltag",
            corner_refinement_win_size=5,
            try_refine_markers=True
        )
        cls.success = cls.calib.detect_charuco()

    def test_detection_success(self):
        """Verify that ChArUco detection succeeds on the test image."""
        self.assertTrue(self.success, "ChArUco detection failed on leftCameraBuffer.png")

    def test_marker_count(self):
        """Verify that the number of detected markers matches the baseline."""
        self.assertEqual(len(self.calib.marker_ids), 16)

    def test_corner_count(self):
        """Verify that the number of detected ChArUco corners matches the baseline."""
        self.assertEqual(len(self.calib.charuco_corners), 22)

    def test_scale_metrics(self):
        """Verify that horizontal and vertical grid scale factors match the baseline."""
        expected_h = 0.011387797
        expected_v = 0.010909438
        self.assertAlmostEqual(self.calib.mm_per_px_h, expected_h, places=5)
        self.assertAlmostEqual(self.calib.mm_per_px_v, expected_v, places=5)

    def test_collinearity_deviation(self):
        """Verify that collinearity distortion metrics match the baseline."""
        expected_max = 6.1987214
        expected_rms = 2.3043513
        self.assertAlmostEqual(self.calib.collin_max, expected_max, places=4)
        self.assertAlmostEqual(self.calib.collin_rms, expected_rms, places=4)

    def test_reprojection_error(self):
        """Verify that reprojection error metrics match the baseline."""
        expected_mean = 11.259812
        expected_max = 23.113803
        self.assertAlmostEqual(self.calib.reproj_mean, expected_mean, places=4)
        self.assertAlmostEqual(self.calib.reproj_max, expected_max, places=4)

    def test_pose_depth(self):
        """Verify that solved pose depth (Z-distance) matches the baseline."""
        expected_depth = 29.302714
        self.assertAlmostEqual(self.calib.pose_depth, expected_depth, places=4)

    def test_camera_matrix(self):
        """Verify that solved camera intrinsics match the baseline pinhole defaults."""
        expected_K = np.array([
            [2592.0, 0.0, 1296.0],
            [0.0, 2592.0, 972.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_allclose(self.calib.K, expected_K, rtol=1e-5)

    def test_results_serialization_and_csv(self):
        """Verify results dict construction, JSON saving, and CSV saving functions."""
        import tempfile
        import json
        
        # Test to_results_dict
        res_dict = self.calib.to_results_dict(squares_x=7, squares_y=5, square_length=3.5, marker_length=2.0)
        self.assertEqual(res_dict["squares_x"], 7)
        self.assertEqual(res_dict["detected_markers"], 16)
        self.assertEqual(res_dict["detected_corners"], 22)
        
        # Test save_results_json and save_corners_csv using temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "results.json")
            csv_path = os.path.join(tmpdir, "corners.csv")
            
            self.calib.save_results_json(json_path, squares_x=7, squares_y=5, square_length=3.5, marker_length=2.0)
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, "r") as f:
                loaded_json = json.load(f)
            self.assertEqual(loaded_json["detected_corners"], 22)
            
            csv_success = self.calib.save_corners_csv(csv_path)
            self.assertTrue(csv_success)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r") as f:
                lines = f.readlines()
            self.assertEqual(lines[0].strip(), "corner_id,x,y")
            self.assertEqual(len(lines), 23) # Header + 22 corners

    def test_calibration_save_and_print(self):
        """Verify multi-image classmethods save_calibration_json and print_calibration_results."""
        import tempfile
        import json
        
        dummy_results = {
            "reproj_error": 0.15,
            "K": np.eye(3),
            "dist_coeffs": np.zeros(5),
            "rvecs": [np.zeros((3, 1))],
            "tvecs": [np.zeros((3, 1))],
            "image_paths": ["img1.png"],
            "image_size": (640, 480)
        }
        
        # Capture print output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            CameraCalibration.print_calibration_results(dummy_results)
        finally:
            sys.stdout = sys.__stdout__
            
        output_str = captured_output.getvalue()
        self.assertIn("MULTI-IMAGE CALIBRATION RESULTS", output_str)
        self.assertIn("Reprojection Error (RMS):       0.1500 pixels", output_str)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_json_path = os.path.join(tmpdir, "calibration_results.json")
            CameraCalibration.save_calibration_json(dummy_results, calib_json_path)
            self.assertTrue(os.path.exists(calib_json_path))
            with open(calib_json_path, "r") as f:
                data = json.load(f)
            self.assertEqual(data["reproj_error"], 0.15)
            self.assertEqual(data["image_paths"], ["img1.png"])

    def test_single_image_auto_mirror(self):
        """Verify that single image auto-mirroring detects and flips corners back correctly."""
        import tempfile
        import cv2
        
        # 1. Load original image and flip it horizontally
        original_img = cv2.imread(self.image_path)
        flipped_img = cv2.flip(original_img, 1)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            flipped_path = os.path.join(tmpdir, "flipped.png")
            cv2.imwrite(flipped_path, flipped_img)
            
            # 2. Test detect_charuco with auto_mirror=False (should fail or find less corners)
            calib_no_mirror = CameraCalibration(
                image_path=flipped_path,
                squares_x=7,
                squares_y=5,
                square_length=3.5,
                marker_length=2.0,
                corner_refinement_method="apriltag",
                try_refine_markers=True
            )
            success_no_mirror = calib_no_mirror.detect_charuco(auto_mirror=False)
            # Flipped image has mirrored markers; ArUco dictionary is not mirror-invariant.
            # Thus, ArUco detection should detect 0 markers and 0 corners.
            self.assertFalse(success_no_mirror)
            self.assertTrue(calib_no_mirror.charuco_corners is None or len(calib_no_mirror.charuco_corners) == 0)
            
            # 3. Test detect_charuco with auto_mirror=True (should succeed by internally flipping)
            calib_mirror = CameraCalibration(
                image_path=flipped_path,
                squares_x=7,
                squares_y=5,
                square_length=3.5,
                marker_length=2.0,
                corner_refinement_method="apriltag",
                try_refine_markers=True
            )
            success_mirror = calib_mirror.detect_charuco(auto_mirror=True)
            self.assertTrue(success_mirror)
            self.assertEqual(len(calib_mirror.charuco_corners), 22)
            self.assertEqual(len(calib_mirror.marker_ids), 16)
            
            # 4. Assert that mapped-back corner coords match the mirrored original coordinates
            # For each corner, the mirrored original coordinate is (w - 1 - x_orig)
            h, w = original_img.shape[:2]
            for i in range(len(self.calib.charuco_ids)):
                cid = self.calib.charuco_ids[i][0]
                # Find matching corner in calib_mirror
                idx_mirror = np.where(calib_mirror.charuco_ids == cid)[0][0]
                
                orig_pt = self.calib.charuco_corners[i][0]
                mirror_pt = calib_mirror.charuco_corners[idx_mirror][0]
                
                # The x-coordinate in mirror_pt must be w - 1 - orig_pt[x]
                expected_x = w - 1 - orig_pt[0]
                self.assertAlmostEqual(mirror_pt[0], expected_x, delta=1.0)
                # y-coordinate should be identical
                self.assertAlmostEqual(mirror_pt[1], orig_pt[1], delta=1.0)

if __name__ == "__main__":
    unittest.main()


