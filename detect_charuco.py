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
COLOR_TEXT = (0, 0, 0)
COLOR_TITLE = (0, 0, 0)

def draw_detected_markers_custom(image, corners, ids, line_thickness=5, font_scale=1.5, text_thickness=3):
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
        # Black outline
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        # Green text
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_GREEN, text_thickness, cv2.LINE_AA)

def draw_detected_corners_charuco_custom(image, corners, ids, corner_radius=10, line_thickness=3, font_scale=1.2, text_thickness=2):
    """Draws detected Charuco chessboard corners with custom sizes and custom text labels."""
    if ids is None or corners is None:
        return
    for i, corner_id in enumerate(ids):
        pt = tuple(corners[i][0].astype(np.int32))
        cid_val = corner_id[0]
        
        # Draw a custom target crosshair
        # Red outer ring
        cv2.circle(image, pt, corner_radius, COLOR_RED, line_thickness, cv2.LINE_AA)
        # Inner dot
        cv2.circle(image, pt, 2, COLOR_RED, -1, cv2.LINE_AA)
        # Crosshair lines
        cv2.line(image, (pt[0] - corner_radius - 4, pt[1]), (pt[0] + corner_radius + 4, pt[1]), COLOR_RED, line_thickness)
        cv2.line(image, (pt[0], pt[1] - corner_radius - 4), (pt[0], pt[1] + corner_radius + 4), COLOR_RED, line_thickness)
        
        # Write corner ID text with a black outline for contrast
        text = str(cid_val)
        text_org = (pt[0] + corner_radius + 6, pt[1] - corner_radius - 2)
        # Black outline
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        # Yellow text
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_YELLOW, text_thickness, cv2.LINE_AA)

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
        

class CameraCalibration:
    """Combines board detection, pose/distortion solving, and image undistortion."""
    def __init__(self, image_path, squares_x=5, squares_y=5, square_length=4.0, marker_length=3.0, dictionary_id=cv2.aruco.DICT_4X4_250):
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
        
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.dictionary)
        
        # Detector Parameters (uses subpixel refinement by default)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(self.board, detectorParams=self.detector_params)
        
        # Class Members for results
        self.charuco_corners = None
        self.charuco_ids = None
        self.marker_corners = None
        self.marker_ids = None
        
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

    def detect_charuco(self, focal_length=None):
        """Performs board detection, pose estimation, scale calculation, and distortion solving using class image self.img."""
        h, w = self.h, self.w
        
        # A. Detect Board
        self.charuco_corners, self.charuco_ids, self.marker_corners, self.marker_ids = self.detector.detectBoard(self.img)
        
        num_markers = len(self.marker_ids) if self.marker_ids is not None else 0
        num_corners = len(self.charuco_corners) if self.charuco_corners is not None else 0
        
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
            
            # Solve Pose
            success, rvec_solved, tvec_solved = cv2.solvePnP(obj_points, img_points, self.K.astype(np.float32), self.dist_coeffs.astype(np.float32))
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
                print("Pose estimation (solvePnP) failed.")
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
            draw_detected_markers_custom(
                annotated_img, 
                self.marker_corners, 
                self.marker_ids, 
                line_thickness=marker_line_width, 
                font_scale=marker_font_size, 
                text_thickness=marker_text_thickness
            )
        if self.charuco_corners is not None:
            draw_detected_corners_charuco_custom(
                annotated_img, 
                self.charuco_corners, 
                self.charuco_ids, 
                corner_radius=corner_radius, 
                line_thickness=corner_line_width, 
                font_scale=corner_font_size, 
                text_thickness=corner_text_thickness
            )
        if self.rvec is not None:
            cv2.drawFrameAxes(annotated_img, self.K.astype(np.float32), np.zeros(5, dtype=np.float32), self.rvec.astype(np.float32), self.tvec.astype(np.float32), length=self.square_length * 2.0, thickness=3)

        # Apply Text Overlay to detected image
        overlay_text_info(
            annotated_img, self.w, self.h, len(self.marker_ids) if self.marker_ids is not None else 0, len(self.charuco_corners) if self.charuco_corners is not None else 0,
            mm_per_px_h=self.mm_per_px_h, mm_per_px_v=self.mm_per_px_v,
            px_per_mm_h=self.px_per_mm_h, px_per_mm_v=self.px_per_mm_v,
            pose_depth=self.pose_depth, euler_xyz=self.euler_xyz,
            collin_max=self.collin_max, reproj_mean=self.reproj_mean,
            solved_dist_coeffs=self.dist_coeffs,
            font_scale=0.9, thickness=2
        )
        return annotated_img

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
        
        if crop and roi is not None:
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
            processed_titled = add_title(processed_img_resized, "Undistorted & Rectified View" if rectify else "Undistorted View")
            
            # Stack side-by-side
            comparison_img = np.hstack((annotated_titled, processed_titled))
            return processed_img, comparison_img
            
        return processed_img

def main():
    parser = argparse.ArgumentParser(description="Object-oriented Charuco Board Detection and Image Undistortion pipeline.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--squares_x", type=int, default=5, help="Number of squares along X-axis (default: 5)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of squares along Y-axis (default: 5)")
    parser.add_argument("--square_length", type=float, default=4.0, help="Square length in mm (default: 4.0)")
    parser.add_argument("--marker_length", type=float, default=3.0, help="Marker length in mm (default: 3.0)")
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
    
    args = parser.parse_args()

    # Load Image
    image_path = args.image_path

    # 1. Initialize and Run Camera Calibration Pipeline
    try:
        calib = CameraCalibration(
            image_path=image_path,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_length=args.square_length,
            marker_length=args.marker_length
        )
    except Exception as e:
        print(e)
        sys.exit(1)
        
    print(f"\nLoaded raw image: '{calib.image_path}' ({calib.w} x {calib.h} pixels)")
    
    success = calib.detect_charuco(focal_length=args.focal_length)
    if not success:
        print("Pipeline aborted due to detection or solving failure.")
        # sys.exit(1)
        
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

    # 3. Apply Undistortion and Rectification using Class Methods (returns rectified + side-by-side comparison)
    undist_only_img = calib.undistort(rectify=False, dist_coeffs=calib.dist_coeffs, crop=args.crop)
    undist_rectified_img, comparison_img = calib.undistort(
        rectify=True, 
        rvec=calib.rvec, 
        dist_coeffs=calib.dist_coeffs, 
        return_comparison=True,
        crop=args.crop
    )

    # 4. Save All Outputs
    input_dir, input_name = os.path.split(image_path)
    base_name, ext = os.path.splitext(input_name)
    
    out_dir = "processed" if os.path.isdir("processed") else input_dir
    
    path_detected = os.path.join(out_dir, f"{base_name}_detected{ext}")
    path_undist = os.path.join(out_dir, f"{base_name}_undistorted{'_cropped' if args.crop else ''}{ext}")
    path_rectified = os.path.join(out_dir, f"{base_name}_undistorted_rectified{'_cropped' if args.crop else ''}{ext}")
    path_comparison = os.path.join(out_dir, f"{base_name}_comparison{'_cropped' if args.crop else ''}{ext}")
    
    cv2.imwrite(path_detected, annotated_img)
    cv2.imwrite(path_undist, undist_only_img)
    cv2.imwrite(path_rectified, undist_rectified_img)
    cv2.imwrite(path_comparison, comparison_img)
    
    print(f"\nSaved annotated image: '{path_detected}'")
    print(f"Saved undistorted image: '{path_undist}'")
    print(f"Saved rectified image:   '{path_rectified}'")
    print(f"Saved comparison image:  '{path_comparison}'")
    print("Combined Pipeline executed successfully!")

if __name__ == "__main__":
    main()
