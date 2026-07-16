import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import tempfile
import sys
import acquire_and_calibrate

class TestAcquireAndCalibrate(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Read the test image to return as mock bytes
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.img_path = os.path.join(test_dir, "data", "leftCameraBuffer.png")
        with open(self.img_path, 'rb') as f:
            self.img_bytes = f.read()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('urllib.request.urlopen')
    @patch('time.sleep')
    def test_acquire_and_calibrate_success(self, mock_sleep, mock_urlopen):
        # Configure urllib mock to return the test image bytes
        mock_response = MagicMock()
        mock_response.read.return_value = self.img_bytes
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        # Prepare arguments: URL, 3 images, output folder, interval=0.0
        test_args = [
            "acquire_and_calibrate.py",
            "http://mock-camera/live",
            "-n", "3",
            "-o", self.temp_dir,
            "--interval", "0.0",
            "--squares_x", "7",
            "--squares_y", "5",
            "--square_length", "3.5",
            "--marker_length", "2.0"
        ]

        with patch.object(sys, 'argv', test_args):
            # Running main should successfully collect 3 images and calibrate
            acquire_and_calibrate.main()

        # Check that calibration_results.json was created
        results_json_path = os.path.join(self.temp_dir, "calibration_results.json")
        self.assertTrue(os.path.exists(results_json_path), "calibration_results.json was not created")

        # Verify 3 images were saved
        for i in range(3):
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{i}.png")))

if __name__ == "__main__":
    unittest.main()
