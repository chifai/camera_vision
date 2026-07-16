import unittest
import numpy as np
import cv2
import arucoUtilities

class TestArucoUtilities(unittest.TestCase):
    def test_to_dict(self):
        """Verify toDict returns correct PredefinedDictionary."""
        d = arucoUtilities.toDict('DICT_4X4_250')
        self.assertIsNotNone(d)
        # Should be an instance of cv2.aruco.Dictionary
        self.assertTrue(hasattr(d, 'bytesList'))

    def test_marker_width(self):
        """Verify markerWidth calculations with/without border."""
        self.assertEqual(arucoUtilities.markerWidth('DICT_4X4_250', include_boarder=True), 6)
        self.assertEqual(arucoUtilities.markerWidth('DICT_5X5_50', include_boarder=False), 5)
        self.assertEqual(arucoUtilities.markerWidth('DICT_7X7_1000', include_boarder=True), 9)

    def test_get_markers(self):
        """Verify getMarkers generates correct image arrays."""
        d = arucoUtilities.toDict('DICT_4X4_250')
        ids = np.array([0, 1])
        pxpm = 100
        markers = arucoUtilities.getMarkers(ids, d, pxpm)
        
        self.assertEqual(len(markers), 2)
        # Each marker image should be pxpm x pxpm numpy array
        self.assertEqual(markers[0].shape, (100, 100))
        self.assertEqual(markers[1].shape, (100, 100))
        self.assertEqual(markers[0].dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
