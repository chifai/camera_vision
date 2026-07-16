import unittest
import os
import shutil
import tempfile
import generate_calib_pic

class TestGenerateCalibPic(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_charuco_to_svg(self):
        """Verify charuco2svg class generates SVG file successfully."""
        svg_path = os.path.join(self.temp_dir, "board.svg")
        generator = generate_calib_pic.charuco2svg(
            SQUARE_X=5,
            SQUARE_Y=4,
            SQUARE_LENGTH=0.04,
            MARKER_LENGTH=0.03,
            DICT_STRING="DICT_4X4_250",
            SVG_PATH=svg_path
        )
        generator.generateSVG()
        self.assertTrue(os.path.exists(svg_path))
        self.assertGreater(os.path.getsize(svg_path), 0)

    def test_generate_charuco_png_helper(self):
        """Verify generate_charuco_png helper function generates PNG file successfully."""
        png_path = os.path.join(self.temp_dir, "board.png")
        generate_calib_pic.generate_charuco_png(
            squares_x=5,
            squares_y=4,
            square_length=0.04,
            marker_length=0.03,
            dict_string="DICT_4X4_250",
            png_path=png_path,
            dpi=100,
            margin_px=5
        )
        self.assertTrue(os.path.exists(png_path))
        self.assertGreater(os.path.getsize(png_path), 0)

if __name__ == "__main__":
    unittest.main()
