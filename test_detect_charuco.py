import unittest
import os
import numpy as np
from detect_charuco import CameraCalibration

class TestCharucoDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_path = os.path.join("raw", "leftCameraBuffer.png")
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

if __name__ == "__main__":
    unittest.main()
