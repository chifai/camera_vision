import unittest
import os
import sys
import shutil
import tempfile
import cv2
import numpy as np
import find_edges

class TestFindEdges(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a synthetic image with a strong, clean white rectangle on a black background
        # This guarantees horizontal and vertical edges for Canny/Hough line detection
        self.img_path = os.path.join(self.temp_dir, "test_synthetic.png")
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (600, 450), (255, 255, 255), -1)
        cv2.imwrite(self.img_path, img)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_find_edges_main(self):
        """Verify that find_edges runs successfully on a synthetic image and outputs results."""
        # Set up sys.argv to point to our synthetic image
        test_argv = ["find_edges.py", self.img_path]
        
        # Verify that output image does not exist yet
        expected_out = os.path.join(self.temp_dir, "test_synthetic_edges.png")
        self.assertFalse(os.path.exists(expected_out))
        
        # Run main in find_edges
        with patch_argv(test_argv):
            find_edges.main()
            
        # Verify output edges image was created
        self.assertTrue(os.path.exists(expected_out), f"Output image {expected_out} was not created")
        
        # Ensure we can read the output image and it is not empty
        out_img = cv2.imread(expected_out)
        self.assertIsNotNone(out_img)

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
