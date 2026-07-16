import unittest
import os
import shutil
import tempfile
import sys
import json
import numpy as np
import cv2
import rectify_folder

class TestRectifyFolder(unittest.TestCase):
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
        
        calib_data = {
            "reproj_error": 0.5,
            "K": K.tolist(),
            "dist_coeffs": dist_coeffs.tolist()
        }
        
        self.json_path = os.path.join(self.temp_dir, "calibration_results.json")
        with open(self.json_path, 'w') as f:
            json.dump(calib_data, f, indent=4)
            
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_rectify_folder_run(self):
        """Test that rectify_folder successfully processes images and outputs annotated results to subfolder."""
        test_argv = [
            "rectify_folder.py",
            self.temp_dir,
            "--squares_x", "7",
            "--squares_y", "5",
            "--square_length", "3.5",
            "--marker_length", "2.0",
            "--subfolder_name", "test_subfolder"
        ]
        
        with patch_argv(test_argv):
            rectify_folder.main()
            
        # Verify the subfolder and files were created
        subfolder_path = os.path.join(self.temp_dir, "test_subfolder")
        self.assertTrue(os.path.exists(subfolder_path), "Output subfolder was not created")
        
        raw_rectified_path = os.path.join(subfolder_path, "leftCameraBuffer_rectified.png")
        annotated_path = os.path.join(subfolder_path, "leftCameraBuffer_rectified_annotated.png")
        results_json_path = os.path.join(subfolder_path, "leftCameraBuffer_results.json")
        corners_csv_path = os.path.join(subfolder_path, "leftCameraBuffer_corners.csv")
        
        self.assertTrue(os.path.exists(raw_rectified_path), "Raw rectified image was not created")
        self.assertTrue(os.path.exists(annotated_path), "Annotated image was not created")
        self.assertTrue(os.path.exists(results_json_path), "Results JSON was not created")
        self.assertTrue(os.path.exists(corners_csv_path), "Corners CSV was not created")
        
        # Ensure we can read the generated images
        self.assertIsNotNone(cv2.imread(raw_rectified_path))
        self.assertIsNotNone(cv2.imread(annotated_path))

class patch_argv:
    """Helper context manager to temporarily override sys.argv."""
    def __init__(self, new_argv):
        self.new_argv = new_argv
        self.old_argv = None

    def __enter__(self):
        self.old_argv = sys.argv
        sys.argv = self.new_argv

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.argv = self.old_argv

if __name__ == "__main__":
    unittest.main()
