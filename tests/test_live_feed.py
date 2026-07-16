import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import cv2
import numpy as np
import live_feed

class TestLiveFeed(unittest.TestCase):
    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.destroyAllWindows')
    @patch('cv2.waitKey')
    def test_live_feed_loop_once(self, mock_wait_key, mock_destroy, mock_named_win, mock_imshow, mock_video_capture):
        # 1. Setup mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        # Load a real image to return
        test_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(test_dir, "data", "leftCameraBuffer.png")
        self.assertTrue(os.path.exists(img_path))
        test_frame = cv2.imread(img_path)
        self.assertIsNotNone(test_frame)
        
        # cap.read() returns (ret, frame)
        mock_cap.read.return_value = (True, test_frame)
        mock_cap.get.return_value = 2592 # mock width/height properties
        mock_video_capture.return_value = mock_cap
        
        # 2. Setup mock waitKey to return 'q' to exit immediately after one loop iteration
        mock_wait_key.return_value = ord('q')
        
        # 3. Patch sys.argv to simulate CLI args
        test_args = [
            "live_feed.py",
            "--index", "0",
            "--squares_x", "7",
            "--squares_y", "5",
            "--square_length", "3.5",
            "--marker_length", "2.0"
        ]
        
        with patch.object(sys, 'argv', test_args):
            live_feed.main()
            
        # 4. Verify mocks were called
        mock_video_capture.assert_called_once()
        mock_cap.read.assert_called()
        mock_imshow.assert_called()
        mock_wait_key.assert_called()
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

if __name__ == "__main__":
    unittest.main()
