import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Create mock PIL and PIL.Image modules and add them to sys.modules
# so we can import and test convert_to_png.py without a hard dependency on Pillow.
mock_image_class = MagicMock()
mock_pil = MagicMock()
mock_pil.Image = mock_image_class

sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_image_class

import convert_to_png

class TestConvertToPng(unittest.TestCase):
    def setUp(self):
        mock_image_class.reset_mock()

    @patch('os.path.exists')
    def test_convert_to_png_success(self, mock_exists):
        # 1. Setup mocks
        mock_exists.return_value = True
        
        mock_img = MagicMock()
        mock_img.mode = 'RGB'
        
        # Configure context manager for Image.open
        mock_image_class.open.return_value = mock_img
        mock_img.__enter__.return_value = mock_img
        
        # 2. Run conversion
        convert_to_png.convert_to_png("dummy.jpg", "dummy_out.png")
        
        # 3. Assertions
        mock_image_class.open.assert_called_once_with("dummy.jpg")
        mock_img.save.assert_called_once_with("dummy_out.png", 'PNG')
        mock_img.convert.assert_not_called() # Since mode is 'RGB', convert('RGB') shouldn't be called

    @patch('os.path.exists')
    def test_convert_to_png_cmyk_conversion(self, mock_exists):
        mock_exists.return_value = True
        
        mock_img = MagicMock()
        mock_img.mode = 'CMYK'
        
        mock_converted_img = MagicMock()
        mock_img.convert.return_value = mock_converted_img
        
        mock_image_class.open.return_value = mock_img
        mock_img.__enter__.return_value = mock_img
        mock_converted_img.__enter__.return_value = mock_converted_img
        
        convert_to_png.convert_to_png("dummy.jpg", "dummy_out.png")
        
        mock_img.convert.assert_called_once_with('RGB')
        mock_converted_img.save.assert_called_once_with("dummy_out.png", 'PNG')

if __name__ == "__main__":
    unittest.main()
