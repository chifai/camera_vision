#!/usr/bin/env python3
"""
Combined Charuco Board Detection & Image Undistortion Script

This script defines the `CameraCalibration` class to detect a Charuco board,
store its calibration/orientation/distortion parameters, and undistort/rectify
images using the solved parameters.

Usage:
    python3 detect_charuco.py <path_to_image> [options]

Author: Antigravity Code Assistant
Date: July 3, 2026
"""

import os
import sys
import argparse
import csv
import datetime
import json
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# Class Constants for Drawing Defaults
FONT_SCALE_DEFAULT = 0.9
LINE_WIDTH_DEFAULT = 2
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_TEXT = COLOR_YELLOW
COLOR_TITLE = COLOR_GREEN

# Canonical mapping of dictionary name strings to OpenCV ArUco constant IDs.
# Importable by other scripts so the definition lives in exactly one place.
DICT_MAPPING = {
    'DICT_4X4_50':    cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100':   cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250':   cv2.aruco.DICT_4X4_250,
    'DICT_4X4_1000':  cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50':    cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100':   cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250':   cv2.aruco.DICT_5X5_250,
    'DICT_5X5_1000':  cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50':    cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100':   cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250':   cv2.aruco.DICT_6X6_250,
    'DICT_6X6_1000':  cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50':    cv2.aruco.DICT_7X7_50,
    'DICT_7X7_100':   cv2.aruco.DICT_7X7_100,
    'DICT_7X7_250':   cv2.aruco.DICT_7X7_250,
    'DICT_7X7_1000':  cv2.aruco.DICT_7X7_1000,
}

class CameraCalibration:
    """Combines board detection, pose/distortion solving, and image undistortion."""

    # ------------------------------------------------------------------
    # Public drawing helpers (usable by external callers directly)
    # ------------------------------------------------------------------

    @staticmethod
    def draw_detected_markers(image, corners, ids, line_thickness=5, font_scale=1.5, text_thickness=3):
        """Draws detected ArUco markers with custom line width and larger, highly visible font."""
        if ids is None or corners is None:
            return
        for i, marker_id in enumerate(ids):
            pts = corners[i][0].astype(np.int32)
            # Draw the 4 border lines of the marker
            for j in range(4):
                cv2.line(image, tuple(pts[j]), tuple(pts[(j + 1) % 4]), COLOR_GREEN, line_thickness)
            # Draw a small red circle on the first corner (top-left by convention)
            cv2.circle(image, tuple(pts[0]), line_thickness + 2, COLOR_RED, -1)
            # Write marker ID text with a black outline for high contrast/legibility
            text = f"id={marker_id[0]}"
            text_org = (int(pts[0][0]) - 15, int(pts[0][1]) - 15)
            cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
            cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_GREEN, text_thickness, cv2.LINE_AA)

    # Backward-compatible alias (name-mangled form used by live_feed.py)
    _CameraCalibration__draw_detected_markers_custom = draw_detected_markers.__func__

    @staticmethod
    def draw_detected_corners(image, corners, ids, corner_radius=10, line_thickness=3, font_scale=1.2, text_thickness=2):
        """Draws detected Charuco chessboard corners with crosshair targets and ID labels."""
        if ids is None or corners is None:
            return
        for i, corner_id in enumerate(ids):
            pt = tuple(corners[i][0].astype(np.int32))
            cid_val = corner_id[0]
            # Red outer ring + inner dot + crosshair
            cv2.circle(image, pt, corner_radius, COLOR_RED, line_thickness, cv2.LINE_AA)
            cv2.circle(image, pt, 2, COLOR_RED, -1, cv2.LINE_AA)
            cv2.line(image, (pt[0] - corner_radius - 4, pt[1]), (pt[0] + corner_radius + 4, pt[1]), COLOR_RED, line_thickness)
            cv2.line(image, (pt[0], pt[1] - corner_radius - 4), (pt[0], pt[1] + corner_radius + 4), COLOR_RED, line_thickness)
            # Yellow corner ID label with black outline
            text = str(cid_val)
            text_org = (pt[0] + corner_radius + 6, pt[1] - corner_radius - 2)
            cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
            cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_YELLOW, text_thickness, cv2.LINE_AA)

    # Backward-compatible alias
    _CameraCalibration__draw_detected_corners_charuco_custom = draw_detected_corners.__func__

    @staticmethod
    def overlay_text_info(
        image, w, h, num_markers, num_corners,
        mm_per_px_h=None, mm_per_px_v=None, px_per_mm_h=None, px_per_mm_v=None,
        pose_depth=None, euler_xyz=None, collin_max=None, reproj_mean=None,
        solved_dist_coeffs=None,
        font_scale=0.9, thickness=2, color=COLOR_TEXT
    ):
        """Overlays calculated parameters (orientation, scale, distortion) as text on the image."""
        y_offset = 40
        line_height = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Charuco Board Analysis", (30, y_offset), font, 1.1, COLOR_TITLE, 3)
        y_offset += 45
        cv2.putText(image, f"Resolution: {w}x{h} px", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        cv2.putText(image, f"Markers: {num_markers} | Corners: {num_corners}", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        if mm_per_px_h is not None and mm_per_px_v is not None:
            cv2.putText(image, f"Scale (X): {mm_per_px_h:.5f} mm/px ({px_per_mm_h:.1f} px/mm)", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"Scale (Y): {mm_per_px_v:.5f} mm/px ({px_per_mm_v:.1f} px/mm)", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        if euler_xyz is not None:
            cv2.putText(image, f"Pose Z (depth): {pose_depth:.1f} mm", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"Roll (Z-rot):  {euler_xyz[2]:+.2f} deg", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"Pitch (X-rot): {euler_xyz[0]:+.2f} deg", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"Yaw (Y-rot):   {euler_xyz[1]:+.2f} deg", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        if collin_max is not None:
            cv2.putText(image, f"Max Line Distort: {collin_max:.2f} px", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        if reproj_mean is not None:
            cv2.putText(image, f"Mean Reproj Error: {reproj_mean:.2f} px", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
        if solved_dist_coeffs is not None:
            cv2.putText(image, f"k1 (radial 1): {solved_dist_coeffs[0]:+.3e}", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"k2 (radial 2): {solved_dist_coeffs[1]:+.3e}", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"p1 (tang 1):   {solved_dist_coeffs[2]:+.3e}", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"p1 (tang 2):   {solved_dist_coeffs[3]:+.3e}", (30, y_offset), font, font_scale, color, thickness)
            y_offset += line_height
            cv2.putText(image, f"k3 (radial 3):   {solved_dist_coeffs[4]:+.3e}", (30, y_offset), font, font_scale, color, thickness)

    # Backward-compatible alias
    _CameraCalibration__overlay_text_info = overlay_text_info.__func__

    @staticmethod
    def create_detector(board, corner_refinement_method="subpix", corner_refinement_win_size=5, try_refine_markers=False,
                        min_marker_perimeter_rate=None, adaptive_thresh_constant=None,
                        adaptive_thresh_win_size_min=None, adaptive_thresh_win_size_max=None, adaptive_thresh_win_size_step=None):
        detector_params = cv2.aruco.DetectorParameters()
        
        methods_map = {
            "none": cv2.aruco.CORNER_REFINE_NONE,
            "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
            "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
            "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG
        }
        # handle case where corner_refinement_method might be integer already
        if isinstance(corner_refinement_method, int):
            detector_params.cornerRefinementMethod = corner_refinement_method
        else:
            method_val = methods_map.get(corner_refinement_method.lower(), cv2.aruco.CORNER_REFINE_SUBPIX)
            detector_params.cornerRefinementMethod = method_val
            
        detector_params.cornerRefinementWinSize = corner_refinement_win_size
        
        if min_marker_perimeter_rate is not None:
            detector_params.minMarkerPerimeterRate = min_marker_perimeter_rate
        if adaptive_thresh_constant is not None:
            detector_params.adaptiveThreshConstant = adaptive_thresh_constant
        if adaptive_thresh_win_size_min is not None:
            detector_params.adaptiveThreshWinSizeMin = adaptive_thresh_win_size_min
        if adaptive_thresh_win_size_max is not None:
            detector_params.adaptiveThreshWinSizeMax = adaptive_thresh_win_size_max
        if adaptive_thresh_win_size_step is not None:
            detector_params.adaptiveThreshWinSizeStep = adaptive_thresh_win_size_step

        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = try_refine_markers

        detector = cv2.aruco.CharucoDetector(board, charucoParams=charuco_params, detectorParams=detector_params)
        return detector, detector_params, charuco_params

    def __init__(self, image_path, squares_x=5, squares_y=5, square_length=4.0, marker_length=3.0, dictionary_id=cv2.aruco.DICT_4X4_250,
                 corner_refinement_method="subpix", corner_refinement_win_size=5, try_refine_markers=False,
                 min_marker_perimeter_rate=None, adaptive_thresh_constant=None,
                 adaptive_thresh_win_size_min=None, adaptive_thresh_win_size_max=None, adaptive_thresh_win_size_step=None):
        self.image_path = image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: Image file '{image_path}' does not exist.")
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Error: Could not read image at '{image_path}'")
        self.h, self.w = self.img.shape[:2]

        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.dictionary_id = dictionary_id
        
        print(f"param: {squares_x}, {squares_y}, {square_length}, {marker_length}")
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.dictionary)
        
        # Detector Parameters (uses custom or default subpixel refinement)
        self.detector, self.detector_params, self.charuco_params = self.create_detector(
            self.board,
            corner_refinement_method=corner_refinement_method,
            corner_refinement_win_size=corner_refinement_win_size,
            try_refine_markers=try_refine_markers,
            min_marker_perimeter_rate=min_marker_perimeter_rate,
            adaptive_thresh_constant=adaptive_thresh_constant,
            adaptive_thresh_win_size_min=adaptive_thresh_win_size_min,
            adaptive_thresh_win_size_max=adaptive_thresh_win_size_max,
            adaptive_thresh_win_size_step=adaptive_thresh_win_size_step
        )
        
        # Class Members for results
        self.charuco_corners = None
        self.charuco_ids = None
        self.marker_corners = None
        self.marker_ids = None
        self.mirror_state = "none"
        
        self.K = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        
        # Metrics
        self.pose_depth = None
        self.euler_xyz = None
        self.euler_zyx = None
        self.collin_max = None
        self.collin_rms = None
        self.reproj_mean = None
        self.reproj_max = None
        
        # Conversion scale values
        self.mm_per_px_h = None
        self.mm_per_px_v = None
        self.px_per_mm_h = None
        self.px_per_mm_v = None
        self.mm_per_px_pose = None

    def __calculate_collinearity_distortion(self):
        """Fits straight lines to rows/columns and calculates the deviation in pixels."""
        if self.charuco_corners is None or len(self.charuco_corners) < 3:
            return None, None
            
        num_cols = self.squares_x - 1
        row_corners = {}
        col_corners = {}
        
        for i, corner_id in enumerate(self.charuco_ids.flatten()):
            r = corner_id // num_cols
            c = corner_id % num_cols
            pt = self.charuco_corners[i][0]
            
            if r not in row_corners:
                row_corners[r] = []
            row_corners[r].append(pt)
            
            if c not in col_corners:
                col_corners[c] = []
            col_corners[c].append(pt)
            
        all_deviations = []
        
        # Fit lines to rows
        for r, pts in row_corners.items():
            if len(pts) >= 3:
                pts = np.array(pts)
                vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                for pt in pts:
                    d = abs((pt[0] - cx) * vy - (pt[1] - cy) * vx)
                    all_deviations.append(d[0])
                    
        # Fit lines to columns
        for c, pts in col_corners.items():
            if len(pts) >= 3:
                pts = np.array(pts)
                vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
                for pt in pts:
                    d = abs((pt[0] - cx) * vy - (pt[1] - cy) * vx)
                    all_deviations.append(d[0])
                    
        if not all_deviations:
            return None, None
            
        self.collin_max = float(np.max(all_deviations))
        self.collin_rms = float(np.sqrt(np.mean(np.square(all_deviations))))
        return self.collin_max, self.collin_rms

    def detect_charuco(self, focal_length=None, auto_mirror=True):
        """Performs board detection, pose estimation, scale calculation, and distortion solving using class image self.img."""
        h, w = self.h, self.w
        
        # A. Detect Board
        corners, ids, marker_corners, marker_ids = self.detector.detectBoard(self.img)
        num_corners = len(corners) if corners is not None else 0
        best_corners = corners
        best_ids = ids
        best_marker_corners = marker_corners
        best_marker_ids = marker_ids
        
        # Initialize mirror state
        self.mirror_state = "none"
        
        # Check mirrored if auto_mirror is enabled
        if auto_mirror:
            # 1. Try horizontal flip
            flipped_h_img = cv2.flip(self.img, 1)
            flipped_h_corners, flipped_h_ids, flipped_h_marker_corners, flipped_h_marker_ids = self.detector.detectBoard(flipped_h_img)
            n_flipped_h_corners = len(flipped_h_corners) if flipped_h_corners is not None else 0
            
            # 2. Try vertical flip
            flipped_v_img = cv2.flip(self.img, 0)
            flipped_v_corners, flipped_v_ids, flipped_v_marker_corners, flipped_v_marker_ids = self.detector.detectBoard(flipped_v_img)
            n_flipped_v_corners = len(flipped_v_corners) if flipped_v_corners is not None else 0
            
            # Determine which yields the most corners
            if n_flipped_h_corners > num_corners or n_flipped_v_corners > num_corners:
                if n_flipped_h_corners >= n_flipped_v_corners:
                    # HORIZONTAL FLIP
                    if flipped_h_corners is not None:
                        flipped_h_corners[:, 0, 0] = w - 1 - flipped_h_corners[:, 0, 0]
                    
                    if flipped_h_marker_corners is not None:
                        mapped_marker_corners = []
                        for mc in flipped_h_marker_corners:
                            mc_orig = mc.copy()
                            mc_orig[0, :, 0] = w - 1 - mc_orig[0, :, 0]
                            # Re-index to maintain convention (0->1, 1->0, 2->3, 3->2)
                            temp = mc_orig.copy()
                            mc_orig[0, 0] = temp[0, 1]
                            mc_orig[0, 1] = temp[0, 0]
                            mc_orig[0, 2] = temp[0, 3]
                            mc_orig[0, 3] = temp[0, 2]
                            mapped_marker_corners.append(mc_orig)
                        flipped_h_marker_corners = tuple(mapped_marker_corners)
                    
                    best_corners = flipped_h_corners
                    best_ids = flipped_h_ids
                    best_marker_corners = flipped_h_marker_corners
                    best_marker_ids = flipped_h_marker_ids
                    num_corners = n_flipped_h_corners
                    self.mirror_state = "flip_horizontal"
                    print("Using horizontally mirrored detection results to maximize corner count.")
                else:
                    # VERTICAL FLIP
                    if flipped_v_corners is not None:
                        flipped_v_corners[:, 0, 1] = h - 1 - flipped_v_corners[:, 0, 1]
                    
                    if flipped_v_marker_corners is not None:
                        mapped_marker_corners = []
                        for mc in flipped_v_marker_corners:
                            mc_orig = mc.copy()
                            mc_orig[0, :, 1] = h - 1 - mc_orig[0, :, 1]
                            # Re-index to maintain convention (0->3, 1->2, 2->1, 3->0)
                            temp = mc_orig.copy()
                            mc_orig[0, 0] = temp[0, 3]
                            mc_orig[0, 1] = temp[0, 2]
                            mc_orig[0, 2] = temp[0, 1]
                            mc_orig[0, 3] = temp[0, 0]
                            mapped_marker_corners.append(mc_orig)
                        flipped_v_marker_corners = tuple(mapped_marker_corners)
                    
                    best_corners = flipped_v_corners
                    best_ids = flipped_v_ids
                    best_marker_corners = flipped_v_marker_corners
                    best_marker_ids = flipped_v_marker_ids
                    num_corners = n_flipped_v_corners
                    self.mirror_state = "flip_vertical"
                    print("Using vertically mirrored detection results to maximize corner count.")
        
        self.charuco_corners = best_corners
        self.charuco_ids = best_ids
        self.marker_corners = best_marker_corners
        self.marker_ids = best_marker_ids
        
        num_markers = len(self.marker_ids) if self.marker_ids is not None else 0
        
        print(f"Detected {num_markers} ArUco markers.")
        print(f"Detected {num_corners} Charuco corners.")
        
        if num_corners == 0:
            print("No Charuco corners detected. Calibration aborted.")
            return False

        # B. Calculate Image-Plane Pixel-to-mm Conversion Factors
        h_px_dists = []
        v_px_dists = []
        if num_corners > 1:
            for i in range(num_corners):
                id1 = self.charuco_ids[i][0]
                r1, c1 = id1 // (self.squares_x - 1), id1 % (self.squares_x - 1)
                p1 = self.charuco_corners[i][0]
                for j in range(i+1, num_corners):
                    id2 = self.charuco_ids[j][0]
                    r2, c2 = id2 // (self.squares_x - 1), id2 % (self.squares_x - 1)
                    p2 = self.charuco_corners[j][0]
                    
                    if r1 == r2 and abs(c1 - c2) == 1:
                        h_px_dists.append(np.linalg.norm(p1 - p2))
                    if c1 == c2 and abs(r1 - r2) == 1:
                        v_px_dists.append(np.linalg.norm(p1 - p2))
                        
            if h_px_dists:
                avg_h_px = np.mean(h_px_dists)
                self.mm_per_px_h = self.square_length / avg_h_px
                self.px_per_mm_h = avg_h_px / self.square_length
            if v_px_dists:
                avg_v_px = np.mean(v_px_dists)
                self.mm_per_px_v = self.square_length / avg_v_px
                self.px_per_mm_v = avg_v_px / self.square_length

        # C. Calculate Collinearity line straightness deviation
        self.__calculate_collinearity_distortion()

        # D. Pose & Distortion Solving (needs at least 4 coplanar points)
        if num_corners >= 4:
            # Form 3D-2D point correspondences
            all_obj_points = self.board.getChessboardCorners()
            obj_points = []
            img_points = []
            for i in range(num_corners):
                cid = self.charuco_ids[i][0]
                obj_points.append(all_obj_points[cid])
                img_points.append(self.charuco_corners[i][0])
                
            obj_points = np.array(obj_points, dtype=np.float32)
            img_points = np.array(img_points, dtype=np.float32)
            
            # Form virtual pinhole camera matrix K
            f = focal_length if focal_length is not None else float(max(w, h))
            cx = w / 2.0
            cy = h / 2.0
            # self.K = np.array([
            #     [2199.678, 0, 1282.006],
            #     [0, 2203.0518, 959.03865],
            #     [0, 0, 1]
            # ], dtype=np.float64)
            
            self.K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros(5, dtype=np.float64)
            
            # Solve Pose using estimatePoseCharucoBoard
            success, rvec_solved, tvec_solved = cv2.aruco.estimatePoseCharucoBoard(
                self.charuco_corners,
                self.charuco_ids,
                self.board,
                self.K.astype(np.float32),
                self.dist_coeffs.astype(np.float32),
                None,
                None
            )
            if success:
                self.rvec = rvec_solved.astype(np.float64)
                self.tvec = tvec_solved.astype(np.float64)
                self.pose_depth = self.tvec[2][0]
                self.mm_per_px_pose = self.pose_depth / f
                
                # Roll Pitch Yaw Euler Angles
                R, _ = cv2.Rodrigues(self.rvec)
                r_obj = R_scipy.from_matrix(R)
                self.euler_xyz = r_obj.as_euler('xyz', degrees=True)
                self.euler_zyx = r_obj.as_euler('zyx', degrees=True)
                
                # Compute reprojection error under perfect pinhole
                projected_points, _ = cv2.projectPoints(obj_points, self.rvec.astype(np.float32), self.tvec.astype(np.float32), self.K.astype(np.float32), self.dist_coeffs.astype(np.float32))
                reproj_errors = np.linalg.norm(img_points - projected_points.squeeze(), axis=1)
                self.reproj_mean = float(np.mean(reproj_errors))
                self.reproj_max = float(np.max(reproj_errors))
                
                # Solve single-frame lens distortion coefficients by fixing intrinsics (K)
                try:
                    flags_calib = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_K3 
                    calib_K = self.K.copy()
                    calib_dist = np.zeros(5, dtype=np.float64)
                    
                    ret_val, _, solved_dist, _, _ = cv2.calibrateCamera(
                        objectPoints=[obj_points],
                        imagePoints=[img_points],
                        imageSize=(w, h),
                        cameraMatrix=calib_K,
                        distCoeffs=calib_dist,
                        flags=flags_calib
                    )
                    self.dist_coeffs = solved_dist.flatten()
                    print(f"Distortion coefficients solved (RMS: {ret_val:.3f} px).")
                except Exception as e:
                    print(f"Warning: Failed to solve lens distortion coefficients: {e}")
                    self.dist_coeffs = np.zeros(5, dtype=np.float64) # reset to zero if failed
            else:
                print("Pose estimation (estimatePoseCharucoBoard) failed.")
                return False
        else:
            print("Not enough corners for pose estimation (need >= 4).")
            return False

        return True

    def get_annotated_image(self, marker_line_width=5, marker_font_size=1.5, marker_text_thickness=3,
                            corner_radius=10, corner_line_width=3, corner_font_size=1.2, corner_text_thickness=2):
        """Generates and returns the annotated image with all overlays and text info."""
        annotated_img = self.img.copy()
        if self.marker_ids is not None:
            self.draw_detected_markers(
                annotated_img,
                self.marker_corners,
                self.marker_ids,
                line_thickness=marker_line_width,
                font_scale=marker_font_size,
                text_thickness=marker_text_thickness
            )
        if self.charuco_corners is not None:
            self.draw_detected_corners(
                annotated_img,
                self.charuco_corners,
                self.charuco_ids,
                corner_radius=corner_radius,
                line_thickness=corner_line_width,
                font_scale=corner_font_size,
                text_thickness=corner_text_thickness
            )
        if self.rvec is not None:
            cv2.drawFrameAxes(annotated_img, self.K.astype(np.float32), np.zeros(5, dtype=np.float32),
                              self.rvec.astype(np.float32), self.tvec.astype(np.float32),
                              length=self.square_length * 2.0, thickness=3)
        self.overlay_text_info(
            annotated_img, self.w, self.h,
            len(self.marker_ids) if self.marker_ids is not None else 0,
            len(self.charuco_corners) if self.charuco_corners is not None else 0,
            mm_per_px_h=self.mm_per_px_h, mm_per_px_v=self.mm_per_px_v,
            px_per_mm_h=self.px_per_mm_h, px_per_mm_v=self.px_per_mm_v,
            pose_depth=self.pose_depth, euler_xyz=self.euler_xyz,
            collin_max=self.collin_max, reproj_mean=self.reproj_mean,
            solved_dist_coeffs=self.dist_coeffs,
            font_scale=0.9, thickness=2
        )
        return annotated_img

    # ------------------------------------------------------------------
    # Results serialisation helpers
    # ------------------------------------------------------------------

    def to_results_dict(self, image_path=None, squares_x=None, squares_y=None,
                        square_length=None, marker_length=None, timestamp=None):
        """
        Builds a JSON-serialisable dictionary of all computed metrics for this
        detection run.  Pass the optional keyword arguments to include board
        configuration metadata in the dict.
        """
        timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        row_ratios, col_ratios = self.calculate_row_col_ratios()
        data = {
            "timestamp": timestamp,
            "image_path": os.path.abspath(image_path or self.image_path),
            "squares_x": squares_x if squares_x is not None else self.squares_x,
            "squares_y": squares_y if squares_y is not None else self.squares_y,
            "square_length": square_length if square_length is not None else self.square_length,
            "marker_length": marker_length if marker_length is not None else self.marker_length,
            "detected_markers": len(self.marker_ids) if self.marker_ids is not None else 0,
            "detected_corners": len(self.charuco_corners) if self.charuco_corners is not None else 0,
            "mirror_state": getattr(self, "mirror_state", "none"),
            "row_ratios_mm_per_px": {f"row_{r}": ratio for r, ratio in row_ratios.items()},
            "col_ratios_mm_per_px": {f"col_{c}": ratio for c, ratio in col_ratios.items()},
        }
        if self.mm_per_px_h is not None:
            data.update({
                "mm_per_px_h": float(self.mm_per_px_h),
                "mm_per_px_v": float(self.mm_per_px_v),
                "px_per_mm_h": float(self.px_per_mm_h),
                "px_per_mm_v": float(self.px_per_mm_v),
            })
        if self.collin_max is not None:
            data.update({
                "collinear_deviation_max_px": float(self.collin_max),
                "collinear_deviation_rms_px": float(self.collin_rms),
            })
        if self.reproj_mean is not None:
            data.update({
                "reprojection_error_mean_px": float(self.reproj_mean),
                "reprojection_error_max_px": float(self.reproj_max),
            })
        if self.K is not None:
            data.update({
                "K": self.K.tolist(),
                "dist_coeffs": self.dist_coeffs.flatten().tolist() if self.dist_coeffs is not None else [0.0] * 5,
            })
        if self.rvec is not None:
            data.update({
                "rvec": self.rvec.flatten().tolist(),
                "tvec": self.tvec.flatten().tolist(),
                "pose_depth_mm": float(self.pose_depth),
                "euler_xyz_pitch_yaw_roll_deg": self.euler_xyz.tolist() if self.euler_xyz is not None else None,
            })
        return data

    def save_results_json(self, path, **kwargs):
        """Serialises the detection results to *path* as pretty-printed JSON.

        Any extra keyword arguments are forwarded to `to_results_dict` (e.g.
        image_path, squares_x, timestamp …).
        """
        data = self.to_results_dict(**kwargs)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def save_corners_csv(self, path):
        """Writes all detected ChArUco corner pixel coordinates to a CSV file.

        Columns: corner_id, x, y
        Does nothing (and returns False) if no corners are detected.
        """
        if self.charuco_corners is None or self.charuco_ids is None:
            return False
        with open(path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["corner_id", "x", "y"])
            for i in range(len(self.charuco_ids)):
                cid = int(self.charuco_ids[i][0])
                pt = self.charuco_corners[i][0]
                writer.writerow([cid, float(pt[0]), float(pt[1])])
        return True

    # ------------------------------------------------------------------
    # Class-level calibration helpers (multi-image results)
    # ------------------------------------------------------------------

    @classmethod
    def save_calibration_json(cls, results, path):
        """Saves the dict returned by `calibrate_from_directory` to *path* as JSON.

        The saved keys mirror what `batch_undistort.py` expects to load:
        reproj_error, K, dist_coeffs, rvecs, image_size, image_paths.
        """
        data = {
            "reproj_error": float(results["reproj_error"]),
            "K": results["K"].tolist(),
            "dist_coeffs": results["dist_coeffs"].flatten().tolist(),
            "rvecs": [r.flatten().tolist() for r in results["rvecs"]],
            "image_size": results["image_size"],
            "image_paths": [os.path.basename(p) for p in results["image_paths"]],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return data

    @classmethod
    def print_calibration_results(cls, results):
        """Prints a human-readable summary of the dict from `calibrate_from_directory`."""
        print("=" * 59)
        print(" MULTI-IMAGE CALIBRATION RESULTS")
        print("=" * 59)
        print(f"Successfully Calibrated Views: {len(results['image_paths'])}")
        print(f"Reprojection Error (RMS):       {results['reproj_error']:.4f} pixels")
        print("\nSolved Camera Matrix K:")
        print(results["K"])
        print("\nSolved Distortion Coefficients D:")
        print(f"  k1, k2, p1, p2, k3 = {results['dist_coeffs'].flatten().tolist()}")
        print("=" * 59)

    def undistort(self, image=None, rectify=False, rvec=None, dist_coeffs=None, return_comparison=False, crop=False):
        """
        Undistorts the image. If image is None, uses self.img.
        If dist_coeffs is None, no distortion correction is applied.
        If rvec is None or rectify=False, no perspective rectification is applied.
        If crop is True, crops the image using cv2.getOptimalNewCameraMatrix to remove invalid boundary pixels.
        If return_comparison is True, returns a tuple (processed_img, side_by_side_comparison_img).
        """
        img_to_process = image if image is not None else self.img
        h, w = img_to_process.shape[:2]
        
        # Resolve camera matrix K (fallback to default pinhole properties if not solved)
        K = self.K
        if K is None:
            f = float(max(w, h))
            cx = w / 2.0
            cy = h / 2.0
            K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1]
            ], dtype=np.float64)
            
        # If dist_coeffs is None, apply zero distortion (no correction)
        D = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float64)
            
        # Determine rotation for rectification (if rectify is True and rvec is provided)
        R_rect = None
        if rectify and rvec is not None:
            R_mat, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
            R_rect = R_mat.T
                
        # Resolve camera matrix for warping
        K_new = K
        roi = None
        if crop:
            # Calculate optimal camera matrix to crop out black out-of-bounds pixels
            K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)

        map1, map2 = cv2.initUndistortRectifyMap(K, D, R_rect, K_new, (w, h), cv2.CV_32FC1)
        processed_img = cv2.remap(img_to_process, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        if crop:
            if rectify and rvec is not None:
                # Crop to the bounding box of the valid non-white warped area
                gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x_c, y_c, w_c, h_c = cv2.boundingRect(c)
                    if w_c > 0 and h_c > 0:
                        processed_img = processed_img[y_c:y_c+h_c, x_c:x_c+w_c]
            elif roi is not None:
                x, y, w_roi, h_roi = roi
                if w_roi > 0 and h_roi > 0:
                    processed_img = processed_img[y:y+h_roi, x:x+w_roi]

        if return_comparison:
            # Generate annotated image
            annotated_img = self.get_annotated_image()
            
            # If the processed image was cropped, resize it to match the height of annotated_img
            # so they can be stacked side-by-side using np.hstack
            if processed_img.shape[0] != h:
                scale = h / processed_img.shape[0]
                new_w = int(processed_img.shape[1] * scale)
                processed_img_resized = cv2.resize(processed_img, (new_w, h), interpolation=cv2.INTER_AREA)
            else:
                processed_img_resized = processed_img
            
            # Helper to add a centered title on top of each image
            def add_title(img_src, title_text, bar_height=80, font_scale=1.2, thickness=3):
                img_h, img_w = img_src.shape[:2]
                bar = np.zeros((bar_height, img_w, 3), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size, _ = cv2.getTextSize(title_text, font, font_scale, thickness)
                text_w, text_h = text_size
                text_x = (img_w - text_w) // 2
                text_y = (bar_height + text_h) // 2
                cv2.putText(bar, title_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                return np.vstack((bar, img_src))
                
            # Add titles
            annotated_titled = add_title(annotated_img, "Detected & Annotated View")
            is_undistorted = (dist_coeffs is not None and not np.all(dist_coeffs == 0))
            if rectify and is_undistorted:
                processed_title = "Undistorted & Rectified View"
            elif rectify:
                processed_title = "Rectified View"
            elif is_undistorted:
                processed_title = "Undistorted View"
            else:
                processed_title = "Original View"
            processed_titled = add_title(processed_img_resized, processed_title)
            
            # Stack side-by-side
            comparison_img = np.hstack((annotated_titled, processed_titled))
            return processed_img, comparison_img
            
        return processed_img

    def calculate_corners_in_mm(self):
        """
        Reconstructs the 2D coordinates of all detected corners in mm on the board plane.
        Returns a dictionary mapping cid to (X_mm, Y_mm).
        """
        if self.charuco_corners is None or self.charuco_ids is None:
            return {}
            
        pts_px = self.charuco_corners.reshape(-1, 2)
        
        if self.rvec is not None and self.tvec is not None and self.K is not None:
            # Use camera pose for back-projection (undistorted)
            R, _ = cv2.Rodrigues(self.rvec)
            dist_coeffs = self.dist_coeffs if self.dist_coeffs is not None else np.zeros(5)
            pts_undist = cv2.undistortPoints(pts_px.reshape(-1, 1, 2), self.K, dist_coeffs).reshape(-1, 2)
            
            pts_mm = []
            for i in range(len(pts_undist)):
                x_n, y_n = pts_undist[i]
                M = np.zeros((3, 3))
                M[:, 0] = R[:, 0]
                M[:, 1] = R[:, 1]
                M[:, 2] = -np.array([x_n, y_n, 1.0])
                try:
                    res = np.linalg.solve(M, -self.tvec.flatten())
                    pts_mm.append((res[0], res[1]))
                except np.linalg.LinAlgError:
                    # Fallback to homography if singular
                    pts_mm.append((0.0, 0.0))
            pts_mm = np.array(pts_mm)
        else:
            # Use homography
            all_board_corners = self.board.getChessboardCorners()
            obj_pts = np.array([all_board_corners[cid[0]][:2] for cid in self.charuco_ids], dtype=np.float32)
            img_pts = np.array([c[0] for c in self.charuco_corners], dtype=np.float32)
            H, _ = cv2.findHomography(img_pts, obj_pts)
            
            pts_homg = np.hstack([pts_px, np.ones((len(pts_px), 1))])
            pts_mm = (H @ pts_homg.T).T
            pts_mm = pts_mm[:, :2] / pts_mm[:, 2:3]
            
        return {cid: (pt[0], pt[1]) for cid, pt in zip(self.charuco_ids.flatten(), pts_mm)}

    def calculate_row_col_ratios(self):
        """
        Calculates the X-mm/pixel ratio for each row and the Y-mm/pixel ratio for each column.
        Returns two dictionaries: row_ratios (row_idx -> float) and col_ratios (col_idx -> float).
        """
        if self.charuco_corners is None or self.charuco_ids is None:
            return {}, {}
            
        pts_px_map = {cid: tuple(pt) for cid, pt in zip(self.charuco_ids.flatten(), self.charuco_corners.reshape(-1, 2))}
        
        cols = self.squares_x - 1
        rows = self.squares_y - 1
        
        row_ratios = {}
        col_ratios = {}
        
        # Calculate X-mm/pixel ratio for each row
        for r in range(rows):
            ratios_in_row = []
            for c in range(cols - 1):
                cid1 = r * cols + c
                cid2 = r * cols + (c + 1)
                if cid1 in pts_px_map and cid2 in pts_px_map:
                    p1 = np.array(pts_px_map[cid1])
                    p2 = np.array(pts_px_map[cid2])
                    d_px = np.linalg.norm(p1 - p2)
                    if d_px > 0:
                        # ratio is physical mm (square_length) / pixel distance
                        ratios_in_row.append(self.square_length / d_px)
            if ratios_in_row:
                row_ratios[r] = float(np.mean(ratios_in_row))
                
        # Calculate Y-mm/pixel ratio for each column
        for c in range(cols):
            ratios_in_col = []
            for r in range(rows - 1):
                cid1 = r * cols + c
                cid2 = (r + 1) * cols + c
                if cid1 in pts_px_map and cid2 in pts_px_map:
                    p1 = np.array(pts_px_map[cid1])
                    p2 = np.array(pts_px_map[cid2])
                    d_px = np.linalg.norm(p1 - p2)
                    if d_px > 0:
                        ratios_in_col.append(self.square_length / d_px)
            if ratios_in_col:
                col_ratios[c] = float(np.mean(ratios_in_col))
                
        return row_ratios, col_ratios

    def get_corners_distance_plot(self):
        """
        Plots all detected corners with a red dot of 10 pixels radius.
        Connects adjacent corners with lines and displays the distance between them in mm.
        """
        plot_img = self.img.copy()
        if self.charuco_corners is None or self.charuco_ids is None:
            return plot_img
            
        # Get coordinates in mm on the board plane
        pts_mm_map = self.calculate_corners_in_mm()
        
        # Create map of pixel coordinates
        pts_px_map = {cid: tuple(pt) for cid, pt in zip(self.charuco_ids.flatten(), self.charuco_corners.reshape(-1, 2))}
        
        cols = self.squares_x - 1
        rows = self.squares_y - 1
        
        # 1. Plot all corners with a red dot of 10 pixels radius
        for cid, pt in pts_px_map.items():
            center = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(plot_img, center, 10, COLOR_RED, -1, cv2.LINE_AA)
            # Write the corner ID in yellow for visibility
            cv2.putText(plot_img, str(cid), (center[0] + 12, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2, cv2.LINE_AA)

        # 2. Draw connections and write distances in mm
        drawn_connections = set()
        
        for cid in pts_px_map.keys():
            r = cid // cols
            c = cid % cols
            
            # Check right neighbor
            if c + 1 < cols and (cid + 1) in pts_px_map:
                pair = tuple(sorted((cid, cid + 1)))
                if pair not in drawn_connections:
                    drawn_connections.add(pair)
                    p1 = pts_px_map[cid]
                    p2 = pts_px_map[cid + 1]
                    
                    # Calculate distance in mm
                    mm1 = pts_mm_map.get(cid, (0, 0))
                    mm2 = pts_mm_map.get(cid + 1, (0, 0))
                    dist_mm = math.sqrt((mm1[0] - mm2[0])**2 + (mm1[1] - mm2[1])**2)
                    
                    # Draw connection line (green line)
                    pt1 = (int(round(p1[0])), int(round(p1[1])))
                    pt2 = (int(round(p2[0])), int(round(p2[1])))
                    cv2.line(plot_img, pt1, pt2, COLOR_GREEN, 2, cv2.LINE_AA)
                    
                    # Draw text for distance at the midpoint
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    text = f"{dist_mm:.2f}mm"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    thick = 2
                    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
                    
                    # Center the box
                    bg_pt1 = (mid[0] - tw // 2 - 4, mid[1] - th // 2 - 4)
                    bg_pt2 = (mid[0] + tw // 2 + 4, mid[1] + th // 2 + 4)
                    cv2.rectangle(plot_img, bg_pt1, bg_pt2, (0, 0, 0), -1)
                    cv2.putText(plot_img, text, (mid[0] - tw // 2, mid[1] + th // 2), font, scale, COLOR_YELLOW, thick, cv2.LINE_AA)
                    
            # Check bottom neighbor
            if r + 1 < rows and (cid + cols) in pts_px_map:
                pair = tuple(sorted((cid, cid + cols)))
                if pair not in drawn_connections:
                    drawn_connections.add(pair)
                    p1 = pts_px_map[cid]
                    p2 = pts_px_map[cid + cols]
                    
                    # Calculate distance in mm
                    mm1 = pts_mm_map.get(cid, (0, 0))
                    mm2 = pts_mm_map.get(cid + cols, (0, 0))
                    dist_mm = math.sqrt((mm1[0] - mm2[0])**2 + (mm1[1] - mm2[1])**2)
                    
                    # Draw connection line (green line)
                    pt1 = (int(round(p1[0])), int(round(p1[1])))
                    pt2 = (int(round(p2[0])), int(round(p2[1])))
                    cv2.line(plot_img, pt1, pt2, COLOR_GREEN, 2, cv2.LINE_AA)
                    
                    # Draw text for distance at the midpoint
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    text = f"{dist_mm:.2f}mm"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    thick = 2
                    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
                    
                    bg_pt1 = (mid[0] - tw // 2 - 4, mid[1] - th // 2 - 4)
                    bg_pt2 = (mid[0] + tw // 2 + 4, mid[1] + th // 2 + 4)
                    cv2.rectangle(plot_img, bg_pt1, bg_pt2, (0, 0, 0), -1)
                    cv2.putText(plot_img, text, (mid[0] - tw // 2, mid[1] + th // 2), font, scale, COLOR_YELLOW, thick, cv2.LINE_AA)
                    
        # Calculate row and column ratios
        row_ratios, col_ratios = self.calculate_row_col_ratios()
        
        # 3. Display row ratios on the plot (left of the leftmost corner of each row)
        for r, ratio in row_ratios.items():
            row_cids = [r * cols + c for c in range(cols) if (r * cols + c) in pts_px_map]
            if row_cids:
                leftmost_cid = min(row_cids, key=lambda cid: pts_px_map[cid][0])
                pt = pts_px_map[leftmost_cid]
                
                text = f"R{r}: {ratio:.5f} mm/px"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.55
                thick = 2
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
                
                tx = max(10, int(round(pt[0])) - tw - 25)
                ty = int(round(pt[1])) + th // 2
                
                cv2.rectangle(plot_img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
                cv2.putText(plot_img, text, (tx, ty), font, scale, COLOR_GREEN, thick, cv2.LINE_AA)
                
        # 4. Display column ratios on the plot (above the top-most corner of each column)
        for c, ratio in col_ratios.items():
            col_cids = [r * cols + c for r in range(rows) if (r * cols + c) in pts_px_map]
            if col_cids:
                topmost_cid = min(col_cids, key=lambda cid: pts_px_map[cid][1])
                pt = pts_px_map[topmost_cid]
                
                text = f"C{c}: {ratio:.5f} mm/px"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.55
                thick = 2
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
                
                tx = int(round(pt[0])) - tw // 2
                ty = max(th + 10, int(round(pt[1])) - 25)
                
                cv2.rectangle(plot_img, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
                cv2.putText(plot_img, text, (tx, ty), font, scale, COLOR_GREEN, thick, cv2.LINE_AA)
                
        return plot_img

    @classmethod
    def calibrate_from_directory(cls, directory_path, squares_x=5, squares_y=5, square_length=4.0, marker_length=3.0, dictionary_id=cv2.aruco.DICT_4X4_250, auto_mirror=True,
                                 corner_refinement_method="subpix", corner_refinement_win_size=5, try_refine_markers=False,
                                 min_marker_perimeter_rate=None, adaptive_thresh_constant=None,
                                 adaptive_thresh_win_size_min=None, adaptive_thresh_win_size_max=None, adaptive_thresh_win_size_step=None):
        """
        Scans a directory for calibration images, detects Charuco corners, and solves for the camera intrinsics and distortion.
        
        If auto_mirror is True, it checks both original and horizontally mirrored versions to maximize detections.
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"'{directory_path}' is not a valid directory.")

        # Set up dictionary and board for detection
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
        
        detector, detector_params, charuco_params = cls.create_detector(
            board,
            corner_refinement_method=corner_refinement_method,
            corner_refinement_win_size=corner_refinement_win_size,
            try_refine_markers=try_refine_markers,
            min_marker_perimeter_rate=min_marker_perimeter_rate,
            adaptive_thresh_constant=adaptive_thresh_constant,
            adaptive_thresh_win_size_min=adaptive_thresh_win_size_min,
            adaptive_thresh_win_size_max=adaptive_thresh_win_size_max,
            adaptive_thresh_win_size_step=adaptive_thresh_win_size_step
        )
        
        # Grab all image files
        valid_extensions = {".png", ".jpg", ".jpeg"}
        image_files = []
        for f in os.listdir(directory_path):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                image_files.append(os.path.join(directory_path, f))
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} potential calibration images in '{directory_path}'.")
        
        all_corners = []
        all_ids = []
        used_image_paths = []
        image_size = None
        
        for filepath in image_files:
            img = cv2.imread(filepath)
            if img is None:
                print(f"  Warning: Could not read image '{os.path.basename(filepath)}'. Skipping.")
                continue
                
            h, w = img.shape[:2]
            if image_size is None:
                image_size = (w, h)
            elif image_size != (w, h):
                print(f"  Warning: Size mismatch for '{os.path.basename(filepath)}' ({w}x{h} vs expected {image_size[0]}x{image_size[1]}). Skipping.")
                continue
                
            # Perform detection on original image
            corners, ids, marker_corners, marker_ids = detector.detectBoard(img)
            n_corners = len(corners) if corners is not None else 0
            mirror_str = ""
            
            # Check mirrored if auto_mirror is enabled
            if auto_mirror:
                # Try horizontal flip
                flipped_h_img = cv2.flip(img, 1)
                flipped_h_corners, flipped_h_ids, _, _ = detector.detectBoard(flipped_h_img)
                n_flipped_h_corners = len(flipped_h_corners) if flipped_h_corners is not None else 0
                
                # Try vertical flip
                flipped_v_img = cv2.flip(img, 0)
                flipped_v_corners, flipped_v_ids, _, _ = detector.detectBoard(flipped_v_img)
                n_flipped_v_corners = len(flipped_v_corners) if flipped_v_corners is not None else 0
                
                if n_flipped_h_corners > n_corners or n_flipped_v_corners > n_corners:
                    if n_flipped_h_corners >= n_flipped_v_corners:
                        # Map horizontal flip back
                        if flipped_h_corners is not None:
                            flipped_h_corners[:, 0, 0] = w - 1 - flipped_h_corners[:, 0, 0]
                        corners = flipped_h_corners
                        ids = flipped_h_ids
                        n_corners = n_flipped_h_corners
                        mirror_str = " (Mirrored Horizontal)"
                    else:
                        # Map vertical flip back
                        if flipped_v_corners is not None:
                            flipped_v_corners[:, 0, 1] = h - 1 - flipped_v_corners[:, 0, 1]
                        corners = flipped_v_corners
                        ids = flipped_v_ids
                        n_corners = n_flipped_v_corners
                        mirror_str = " (Mirrored Vertical)"
            
            if n_corners >= 4:
                # Proactively check homography to prevent calibration crashes due to degenerate/collinear corner points
                all_obj_points = board.getChessboardCorners()
                obj_pts = np.array([all_obj_points[cid[0]] for cid in ids], dtype=np.float32)
                img_pts = np.array([c[0] for c in corners], dtype=np.float32)
                
                H, _ = cv2.findHomography(obj_pts[:, :2], img_pts)
                if H is None or H.shape != (3, 3):
                    print(f"  Skipping '{os.path.basename(filepath)}': Homography estimation failed (degenerate/collinear points).")
                    continue

                all_corners.append(corners)
                all_ids.append(ids)
                used_image_paths.append(filepath)
                print(f"  Processed '{os.path.basename(filepath)}'{mirror_str}: detected {n_corners} corners.")
            else:
                print(f"  Skipping '{os.path.basename(filepath)}': only {n_corners} corners detected (minimum 4 required).")
                
        print(f"\nSuccessfully collected corners from {len(all_corners)} images.")
        
        if len(all_corners) < 3:
            print("Error: At least 3 valid calibration views are required to perform calibration.")
            return False, {}
            
        cameraMatrix = np.eye(3, dtype=np.float64)
        distCoeffs = np.zeros(5, dtype=np.float64)
        
        print("\nComputing multi-image camera calibration parameters...")
        try:
            retval, K_solved, dist_solved, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners,
                charucoIds=all_ids,
                board=board,
                imageSize=image_size,
                cameraMatrix=cameraMatrix,
                distCoeffs=distCoeffs
            )
        except AttributeError as e:
            if "calibrateCameraCharuco" in str(e):
                print("\nError: cv2.aruco.calibrateCameraCharuco was not found.")
                print("This usually means you are running the standard 'opencv-python' package globally instead of 'opencv-contrib-python'.")
                print("Please run this command using the project's virtual environment instead:")
                print("  ./run_venv.sh python3 detect_charuco.py ./raw/calibrate")
                sys.exit(1)
            else:
                raise e
        
        results = {
            'K': K_solved,
            'dist_coeffs': dist_solved,
            'reproj_error': retval,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'image_paths': used_image_paths,
            'image_size': image_size
        }
        return True, results

def main():
    # region parse argument
    parser = argparse.ArgumentParser(description="Object-oriented Charuco Board Detection and Image Undistortion pipeline.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--squares_x", type=int, default=7, help="Number of squares along X-axis (default: 5)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of squares along Y-axis (default: 5)")
    parser.add_argument("--square_length", type=float, default=3.5, help="Square length in mm (default: 4.0)")
    parser.add_argument("--marker_length", type=float, default=2.0, help="Marker length in mm (default: 3.0)")
    parser.add_argument("--focal_length", type=float, default=None, help="Assumed focal length in pixels.")
    
    # Custom Overlay parameters
    parser.add_argument("--marker_line_width", type=int, default=5, help="Line width for drawing markers (default: 5)")
    parser.add_argument("--marker_font_size", type=float, default=1.5, help="Font scale for marker IDs (default: 1.5)")
    parser.add_argument("--marker_text_thickness", type=int, default=3, help="Font thickness for marker IDs (default: 3)")
    parser.add_argument("--corner_radius", type=int, default=10, help="Radius for corner crosshairs (default: 10)")
    parser.add_argument("--corner_line_width", type=int, default=3, help="Line width for corner crosshairs (default: 3)")
    parser.add_argument("--corner_font_size", type=float, default=1.2, help="Font scale for corner IDs (default: 1.2)")
    parser.add_argument("--corner_text_thickness", type=int, default=2, help="Font thickness for corner IDs (default: 2)")
    parser.add_argument("--crop", action="store_true", help="Crop the output image to remove invalid boundary/edge pixels (uses optimal camera matrix calculation)")
    
    # Detector/refinement tuning parameters
    parser.add_argument("--corner_refinement_method", type=str, default="apriltag", choices=["none", "subpix", "contour", "apriltag"],
                        help="Corner refinement method (default: subpix)")
    parser.add_argument("--corner_refinement_win_size", type=int, default=5,
                        help="Window size for subpixel corner refinement (default: 5)")
    parser.add_argument("--try_refine_markers", action="store_true",
                        help="Try to refine/find missing markers using board layout (default: False)")
    parser.add_argument("--min_marker_perimeter_rate", type=float, default=None,
                        help="Minimum marker perimeter rate (default: None/OpenCV default)")
    parser.add_argument("--adaptive_thresh_constant", type=float, default=None,
                        help="Adaptive threshold constant (default: None/OpenCV default)")
    parser.add_argument("--adaptive_thresh_win_size_min", type=int, default=None,
                        help="Adaptive threshold window size min (default: None/OpenCV default)")
    parser.add_argument("--adaptive_thresh_win_size_max", type=int, default=None,
                        help="Adaptive threshold window size max (default: None/OpenCV default)")
    parser.add_argument("--adaptive_thresh_win_size_step", type=int, default=None,
                        help="Adaptive threshold window size step (default: None/OpenCV default)")
    parser.add_argument("--auto_mirror", action="store_true", default=True,
                        help="Automatically try mirrored image to maximize corner detection (default: True)")
    parser.add_argument("--no_auto_mirror", action="store_false", dest="auto_mirror",
                        help="Disable auto mirroring")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Base directory for output files (default: ./output)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "jpg", "jpeg"],
                        help="Output image format (default: matches input image format)")
    parser.add_argument("--undistort", action="store_true",
                        help="Apply undistortion and perspective rectification in the end (default: False)")
    
    args = parser.parse_args()
    # endregion parse argument

    # Load Image
    image_path = args.image_path

    if os.path.isdir(image_path):
        print(f"\n===========================================================")
        print(f" Starting Multi-Image Calibration from Directory: '{image_path}'")
        print(f"===========================================================")
        success, results = CameraCalibration.calibrate_from_directory(
            directory_path=image_path,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length,
            auto_mirror=args.auto_mirror,
            corner_refinement_method=args.corner_refinement_method,
            corner_refinement_win_size=args.corner_refinement_win_size,
            try_refine_markers=args.try_refine_markers,
            min_marker_perimeter_rate=args.min_marker_perimeter_rate,
            adaptive_thresh_constant=args.adaptive_thresh_constant,
            adaptive_thresh_win_size_min=args.adaptive_thresh_win_size_min,
            adaptive_thresh_win_size_max=args.adaptive_thresh_win_size_max,
            adaptive_thresh_win_size_step=args.adaptive_thresh_win_size_step
        )
        if success:
            CameraCalibration.print_calibration_results(results)
            out_file = os.path.join(image_path, "calibration_results.json")
            CameraCalibration.save_calibration_json(results, out_file)
            print(f"\nSaved calibration results to: '{out_file}'")
        else:
            print("\nCalibration failed or not enough valid images.")
            print("=" * 59)
            sys.exit(1)
        sys.exit(0)

    # 1. Initialize and Run Camera Calibration Pipeline
    try:
        calib = CameraCalibration(
            image_path=image_path,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length,
            corner_refinement_method=args.corner_refinement_method,
            corner_refinement_win_size=args.corner_refinement_win_size,
            try_refine_markers=args.try_refine_markers,
            min_marker_perimeter_rate=args.min_marker_perimeter_rate,
            adaptive_thresh_constant=args.adaptive_thresh_constant,
            adaptive_thresh_win_size_min=args.adaptive_thresh_win_size_min,
            adaptive_thresh_win_size_max=args.adaptive_thresh_win_size_max,
            adaptive_thresh_win_size_step=args.adaptive_thresh_win_size_step
        )
    except Exception as e:
        print(e)
        sys.exit(1)
        
    print(f"\nLoaded raw image: '{calib.image_path}' ({calib.w} x {calib.h} pixels)")
    
    success = calib.detect_charuco(focal_length=args.focal_length, auto_mirror=args.auto_mirror)
    if not success:
        print("Pipeline aborted due to detection or solving failure.")
        sys.exit(1)
        
    # Output detailed prints
    if calib.marker_ids is not None:
        print("\n--- ArUco Marker Coordinates ---")
        for i, mid in enumerate(calib.marker_ids):
            corners = calib.marker_corners[i][0]
            print(f"Marker ID {mid[0]:2d}: Center=({corners.mean(axis=0)[0]:.2f}, {corners.mean(axis=0)[1]:.2f}) px")
            
    if calib.charuco_corners is not None:
        print("\n--- Charuco Corner Coordinates (Subpixel) ---")
        for i, cid in enumerate(calib.charuco_ids):
            corner = calib.charuco_corners[i][0]
            pt3d = calib.board.getChessboardCorners()[cid[0]]
            print(f"Corner ID {cid[0]:2d}: 2D=({corner[0]:.2f}, {corner[1]:.2f}) px | 3D=({pt3d[0]:.1f}, {pt3d[1]:.1f}, {pt3d[2]:.1f}) mm")

    print("\n--- Calibration & Distortion Metrics ---")
    if calib.mm_per_px_h is not None:
        print(f"  Horizontal grid scale: {calib.mm_per_px_h:.6f} mm/px ({calib.px_per_mm_h:.2f} px/mm)")
        print(f"  Vertical grid scale:   {calib.mm_per_px_v:.6f} mm/px ({calib.px_per_mm_v:.2f} px/mm)")
    if calib.collin_max is not None:
        print(f"  Max Collinearity Deviation: {calib.collin_max:.3f} px")
        print(f"  RMS Collinearity Deviation: {calib.collin_rms:.3f} px")
    if calib.reproj_mean is not None:
        print(f"  Mean Reprojection Error:    {calib.reproj_mean:.3f} px")
        print(f"  Max Reprojection Error:     {calib.reproj_max:.3f} px")
    if calib.dist_coeffs is not None:
        print(f"  Lens Distortion (k1, k2, p1, p2, k3):")
        print(f"    [{calib.dist_coeffs[0]:+.6e}, {calib.dist_coeffs[1]:+.6e}, {calib.dist_coeffs[2]:+.6e}, {calib.dist_coeffs[3]:+.6e}, {calib.dist_coeffs[4]:+.6e}]")
    if calib.euler_xyz is not None:
        print(f"  Board Pose Z (depth): {calib.pose_depth:.3f} mm")
        print(f"  Orientation ('xyz' Euler): Pitch={calib.euler_xyz[0]:+.3f}°, Yaw={calib.euler_xyz[1]:+.3f}°, Roll={calib.euler_xyz[2]:+.3f}°")

    # 2. Generate Annotated Image (used for saving and text overlays)
    annotated_img = calib.get_annotated_image(
        marker_line_width=args.marker_line_width,
        marker_font_size=args.marker_font_size,
        marker_text_thickness=args.marker_text_thickness,
        corner_radius=args.corner_radius,
        corner_line_width=args.corner_line_width,
        corner_font_size=args.corner_font_size,
        corner_text_thickness=args.corner_text_thickness
    )

    # 4. Save Outputs and Results JSON/CSV to timestamped folder in output directory
    input_dir, input_name = os.path.split(image_path)
    base_name, input_ext = os.path.splitext(input_name)


    # Determine output format/extension
    if args.format:
        ext = f".{args.format.lower()}"
        if ext == ".jpeg":
            ext = ".jpg"
    else:
        ext = input_ext

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir_name = f"{timestamp}_{base_name}"
    out_dir = os.path.abspath(os.path.join(args.output_dir, out_dir_name))
    os.makedirs(out_dir, exist_ok=True)

    path_detected = os.path.join(out_dir, f"{base_name}_detected{ext}")
    cv2.imwrite(path_detected, annotated_img)
    print(f"\nSaved annotated image: '{path_detected}'")

    corners_plot_img = calib.get_corners_distance_plot()
    path_corners_plot = os.path.join(out_dir, f"{base_name}_corners_plot{ext}")
    cv2.imwrite(path_corners_plot, corners_plot_img)
    print(f"Saved corners plot:      '{path_corners_plot}'")

    if args.undistort:
        undist_only_img = calib.undistort(rectify=False, dist_coeffs=None, crop=args.crop)
        undist_rectified_img, comparison_img = calib.undistort(
            rectify=True, rvec=calib.rvec, dist_coeffs=None,
            return_comparison=True, crop=args.crop
        )
        suffix = "_cropped" if args.crop else ""
        path_undist = os.path.join(out_dir, f"{base_name}_undistorted{suffix}{ext}")
        path_rectified = os.path.join(out_dir, f"{base_name}_undistorted_rectified{suffix}{ext}")
        path_comparison = os.path.join(out_dir, f"{base_name}_comparison{suffix}{ext}")
        cv2.imwrite(path_undist, undist_only_img)
        cv2.imwrite(path_rectified, undist_rectified_img)
        cv2.imwrite(path_comparison, comparison_img)
        print(f"Saved undistorted image: '{path_undist}'")
        print(f"Saved rectified image:   '{path_rectified}'")
        print(f"Saved comparison image:  '{path_comparison}'")
    else:
        rectified_only_img, comparison_img = calib.undistort(
            rectify=True, rvec=calib.rvec, dist_coeffs=None,
            return_comparison=True, crop=args.crop
        )
        suffix = "_cropped" if args.crop else ""
        path_rectified = os.path.join(out_dir, f"{base_name}_rectified{suffix}{ext}")
        path_comparison = os.path.join(out_dir, f"{base_name}_comparison{suffix}{ext}")
        cv2.imwrite(path_rectified, rectified_only_img)
        cv2.imwrite(path_comparison, comparison_img)
        print(f"Saved rectified image:   '{path_rectified}'")
        print(f"Saved comparison image:  '{path_comparison}'")

    # Save results JSON and corners CSV using instance helpers
    json_path = os.path.join(out_dir, f"{base_name}_results.json")
    calib.save_results_json(json_path, image_path=image_path, timestamp=timestamp,
                            squares_x=args.squares_x, squares_y=args.squares_y,
                            square_length=args.square_length, marker_length=args.marker_length)
    print(f"Saved results JSON:      '{json_path}'")

    csv_path = os.path.join(out_dir, f"{base_name}_corners.csv")
    if calib.save_corners_csv(csv_path):
        print(f"Saved corners CSV:       '{csv_path}'")

    # Run detection again on the rectified image and save to a subfolderder
    print(f"\n===========================================================")
    print(f" Starting Second-Pass Detection on Rectified Image")
    print(f"===========================================================")
    
    # 1. Create a subfolder inside the main out_dir
    rectified_out_dir = os.path.join(out_dir, "rectified_detection")
    os.makedirs(rectified_out_dir, exist_ok=True)
    
    # 2. Instantiate a new CameraCalibration object for the rectified image.
    try:
        rectified_calib = CameraCalibration(
            image_path=path_rectified,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length,
            corner_refinement_method=args.corner_refinement_method,
            corner_refinement_win_size=args.corner_refinement_win_size,
            try_refine_markers=args.try_refine_markers,
            min_marker_perimeter_rate=args.min_marker_perimeter_rate,
            adaptive_thresh_constant=args.adaptive_thresh_constant,
            adaptive_thresh_win_size_min=args.adaptive_thresh_win_size_min,
            adaptive_thresh_win_size_max=args.adaptive_thresh_win_size_max,
            adaptive_thresh_win_size_step=args.adaptive_thresh_win_size_step
        )
    except Exception as e:
        print(f"Error loading rectified image: {e}")
        sys.exit(1)
        
    rectified_success = rectified_calib.detect_charuco(focal_length=args.focal_length, auto_mirror=args.auto_mirror)
    if not rectified_success:
        print("Second-pass detection on rectified image failed or not enough corners found.")
    else:
        print(f"Detected {len(rectified_calib.charuco_corners)} corners on rectified image.")
        
        # 3. Output corners plot image
        rectified_corners_plot = rectified_calib.get_corners_distance_plot()
        rectified_base_name, _ = os.path.splitext(os.path.basename(path_rectified))
        rectified_plot_path = os.path.join(rectified_out_dir, f"{rectified_base_name}_corners_plot{ext}")
        cv2.imwrite(rectified_plot_path, rectified_corners_plot)
        print(f"Saved rectified corners plot:  '{rectified_plot_path}'")
        
        # 4. Output corners CSV using instance helper
        rectified_csv_path = os.path.join(rectified_out_dir, f"{rectified_base_name}_corners.csv")
        if rectified_calib.save_corners_csv(rectified_csv_path):
            print(f"Saved rectified corners CSV:   '{rectified_csv_path}'")

        # 5. Output JSON result using instance helper
        rectified_json_path = os.path.join(rectified_out_dir, f"{rectified_base_name}_results.json")
        rectified_calib.save_results_json(
            rectified_json_path,
            image_path=path_rectified,
            timestamp=timestamp,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length
        )
        print(f"Saved rectified results JSON:  '{rectified_json_path}'")

    print("Combined Pipeline executed successfully!")

if __name__ == "__main__":
    main()
