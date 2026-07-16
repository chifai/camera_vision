import unittest
import os
import sys
import shutil
import tempfile
import cv2
import numpy as np
import sobel_finder

class TestSobelFinder(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a larger synthetic image (1200x1200) with a strong, clean white rectangle
        # to ensure the Hough lines detection threshold (500 votes) is exceeded.
        self.img_path = os.path.join(self.temp_dir, "test_synthetic.png")
        img = np.zeros((1200, 1200, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (1100, 1100), (255, 255, 255), -1)
        cv2.imwrite(self.img_path, img)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
        # Clean up processed directory created by sobel_finder in current working directory
        if os.path.exists("processed"):
            shutil.rmtree("processed")

    def test_sobel_finder_main(self):
        """Verify that sobel_finder runs successfully on a synthetic image and outputs results."""
        # Set up sys.argv to point to our synthetic image
        test_argv = ["sobel_finder.py", self.img_path]
        
        # Verify that output directories/files do not exist yet in processed
        self.assertFalse(os.path.exists("processed"))
        
        # Run main in sobel_finder
        with patch_argv(test_argv):
            sobel_finder.main()
            
        # Verify output sobel debug image and edges image were created
        expected_debug = os.path.join("processed", "debug_sobel", "test_synthetic_sobel_debug.png")
        expected_edges = os.path.join("processed", "edges_sobel", "test_synthetic_sobel_edges.png")
        
        self.assertTrue(os.path.exists(expected_debug), f"Sobel debug image was not created at {expected_debug}")
        self.assertTrue(os.path.exists(expected_edges), f"Sobel edges image was not created at {expected_edges}")
        
        # Ensure we can read the output images
        self.assertIsNotNone(cv2.imread(expected_debug))
        self.assertIsNotNone(cv2.imread(expected_edges))

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
