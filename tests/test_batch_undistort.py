import unittest
import os
import shutil
import tempfile
import json
import numpy as np
import cv2
from batch_undistort import batch_undistort

class MockArgs:
    def __init__(self):
        self.squares_x = 7
        self.squares_y = 5
        self.square_length = 3.5
        self.marker_length = 2.0
        self.no_crop = False
        self.corner_refinement_method = "subpix"
        self.try_refine_markers = True

class TestBatchUndistort(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Paths
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.src_image = os.path.join(test_dir, "data", "leftCameraBuffer.png")
        self.dst_image = os.path.join(self.temp_dir, "leftCameraBuffer.png")
        shutil.copy(self.src_image, self.dst_image)
        
        # Create a mock calibration_results.json
        K = np.array([
            [2592.0, 0.0, 1296.0],
            [0.0, 2592.0, 972.0],
            [0.0, 0.0, 1.0]
        ])
        dist_coeffs = np.zeros(5)
        
        # rvec maps base name to rotation
        # image name is leftCameraBuffer.png
        # let's provide a valid rotation vector
        rvec = [0.0, 0.0, 0.0]
        
        calib_data = {
            "reproj_error": 0.5,
            "K": K.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "image_paths": ["leftCameraBuffer.png"],
            "rvecs": [rvec],
            "image_size": [2592, 1944]
        }
        
        self.json_path = os.path.join(self.temp_dir, "calibration_results.json")
        with open(self.json_path, 'w') as f:
            json.dump(calib_data, f, indent=4)
            
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_batch_undistort_run(self):
        """Test that batch_undistort successfully processes files and creates output files."""
        args = MockArgs()
        # Run batch_undistort
        batch_undistort(self.temp_dir, args)
        
        # Verify the files created
        undistorted_path = os.path.join(self.temp_dir, "leftCameraBuffer_undistorted.png")
        corners_plot_path = os.path.join(self.temp_dir, "leftCameraBuffer_undistorted_corners_plot.png")
        
        self.assertTrue(os.path.exists(undistorted_path), "Undistorted image was not created")
        self.assertTrue(os.path.exists(corners_plot_path), "Corners plot image was not created")
        
        # Ensure we can read the generated images
        undist_img = cv2.imread(undistorted_path)
        self.assertIsNotNone(undist_img)
        
        plot_img = cv2.imread(corners_plot_path)
        self.assertIsNotNone(plot_img)

if __name__ == "__main__":
    unittest.main()
