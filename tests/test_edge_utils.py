import unittest
import math
import numpy as np
from edge_utils import polar_to_points, intersect_polar

class TestEdgeUtils(unittest.TestCase):
    def test_polar_to_points(self):
        """Verify polar_to_points conversion for basic lines."""
        # Horizontal line (theta = pi/2, y = 100)
        pt1, pt2 = polar_to_points(100.0, math.pi / 2)
        # y should be close to 100 (integer truncation might cause it to be 99)
        self.assertLessEqual(abs(pt1[1] - 100), 1)
        self.assertLessEqual(abs(pt2[1] - 100), 1)
        # x coordinates should span far apart
        self.assertTrue(abs(pt1[0] - pt2[0]) > 5000)

        # Vertical line (theta = 0, x = 200)
        pt1, pt2 = polar_to_points(200.0, 0.0)
        self.assertLessEqual(abs(pt1[0] - 200), 1)
        self.assertLessEqual(abs(pt2[0] - 200), 1)
        self.assertTrue(abs(pt1[1] - pt2[1]) > 5000)

    def test_intersect_polar(self):
        """Verify polar intersection math for intersecting and parallel lines."""
        # Line 1: y = 150 -> rho = 150, theta = pi/2
        l1 = (150.0, math.pi / 2)
        # Line 2: x = 250 -> rho = 250, theta = 0
        l2 = (250.0, 0.0)
        
        pt = intersect_polar(l1, l2)
        self.assertEqual(pt, (250, 150))
        
        # Parallel lines (y = 150 and y = 300)
        l3 = (300.0, math.pi / 2)
        pt_parallel = intersect_polar(l1, l3)
        self.assertIsNone(pt_parallel)

if __name__ == "__main__":
    unittest.main()
