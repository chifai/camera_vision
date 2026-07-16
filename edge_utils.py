#!/usr/bin/env python3
"""
Edge utility functions shared by find_edges.py and sobel_finder.py.

Provides polar-line helpers for converting between Hough-space (rho, theta)
and Cartesian image coordinates, plus line intersection math.
"""

import math
import numpy as np


def polar_to_points(rho, theta):
    """Converts polar coordinates (rho, theta) to two far-apart points for drawing."""
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
    pt2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
    return pt1, pt2


def intersect_polar(l1, l2):
    """Finds the intersection point of two lines given in polar (rho, theta) form.

    Returns an (x, y) integer tuple, or None if the lines are parallel.
    """
    rho1, theta1 = l1
    rho2, theta2 = l2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ], dtype=np.float64)
    B = np.array([rho1, rho2], dtype=np.float64)
    try:
        res = np.linalg.solve(A, B)
        return int(round(res[0])), int(round(res[1]))
    except np.linalg.LinAlgError:
        return None
