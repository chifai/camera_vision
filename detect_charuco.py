#!/usr/bin/env python3
"""
Charuco Board Detection Script

This script detects a Charuco board in an input image, prints the detected marker
and corner coordinates, calculates the board's orientation (Roll, Pitch, Yaw),
computes pixel-to-millimeter conversion factors, overlays the results on the image,
and saves the annotated image to a file.

Usage:
    python3 detect_charuco.py <path_to_image> [--square_length 4.0] [--marker_length 3.0]

Author: Antigravity Code Assistant
Date: July 2, 2026
"""

import os
import sys
import argparse
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

FONT_SCALE = 0.2
LINE_WIDTH = 1
TEXT_WIDTH = 1

def draw_detected_markers_custom(image, corners, ids, line_thickness=5, font_scale=1.5, text_thickness=3):
    """Draws detected ArUco markers with custom line width and larger, highly visible font."""
    if ids is None or corners is None:
        return
    for i, marker_id in enumerate(ids):
        pts = corners[i][0].astype(np.int32)
        # Draw the 4 border lines of the marker
        for j in range(4):
            cv2.line(image, tuple(pts[j]), tuple(pts[(j + 1) % 4]), (0, 255, 0), line_thickness)
        
        # Draw a small red circle on the first corner (top-left by convention)
        cv2.circle(image, tuple(pts[0]), line_thickness + 2, (0, 0, 255), -1)
        
        # Write marker ID text with a black outline for high contrast/legibility
        text = f"id={marker_id[0]}"
        # Offset the text slightly from the top-left corner
        text_org = (int(pts[0][0]) - 15, int(pts[0][1]) - 15)
        # Black outline
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        # Green text
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)

def draw_detected_corners_charuco_custom(image, corners, ids, corner_radius=10, line_thickness=3, font_scale=1.2, text_thickness=2):
    """Draws detected Charuco chessboard corners with custom sizes and custom text labels."""
    if ids is None or corners is None:
        return
    for i, corner_id in enumerate(ids):
        pt = tuple(corners[i][0].astype(np.int32))
        cid_val = corner_id[0]
        
        # Draw a custom target crosshair
        # Red outer ring
        cv2.circle(image, pt, corner_radius, (0, 0, 255), line_thickness, cv2.LINE_AA)
        # Inner dot
        cv2.circle(image, pt, 2, (0, 0, 255), -1, cv2.LINE_AA)
        # Crosshair lines
        cv2.line(image, (pt[0] - corner_radius - 4, pt[1]), (pt[0] + corner_radius + 4, pt[1]), (0, 0, 255), line_thickness)
        cv2.line(image, (pt[0], pt[1] - corner_radius - 4), (pt[0], pt[1] + corner_radius + 4), (0, 0, 255), line_thickness)
        
        # Write corner ID text with a black outline for contrast
        text = str(cid_val)
        text_org = (pt[0] + corner_radius + 6, pt[1] - corner_radius - 2)
        # Black outline
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness + 2, cv2.LINE_AA)
        # Yellow text (contrasts nicely against markers and the red crosshair)
        cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

def calculate_collinearity_distortion(corners, ids, squares_x):
    """
    Fits straight lines to rows/columns of corners and calculates the deviation in pixels
    to quantify lens distortion.
    """
    if corners is None or len(corners) < 3:
        return None, None
        
    num_cols = squares_x - 1
    row_corners = {}
    col_corners = {}
    
    for i, corner_id in enumerate(ids.flatten()):
        r = corner_id // num_cols
        c = corner_id % num_cols
        pt = corners[i][0]
        
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
        
    max_dev = float(np.max(all_deviations))
    rms_dev = float(np.sqrt(np.mean(np.square(all_deviations))))
    return max_dev, rms_dev

def main():
    parser = argparse.ArgumentParser(description="Detect Charuco board and estimate pose/pixel-to-mm conversion.")
    parser.add_argument("image_path", type=str, help="Path to the input image (e.g. raw/cameraaligner_data/rightcamera_chauro.png)")
    parser.add_argument("--squares_x", type=int, default=5, help="Number of squares along X-axis (default: 5)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of squares along Y-axis (default: 5)")
    parser.add_argument("--square_length", type=float, default=4.0, help="Chessboard square side length in mm (default: 4.0)")
    parser.add_argument("--marker_length", type=float, default=3.0, help="ArUco marker side length in mm (default: 3.0)")
    parser.add_argument("--focal_length", type=float, default=None, help="Assumed focal length in pixels. Defaults to max(width, height)")
    parser.add_argument("--marker_line_width", type=int, default=LINE_WIDTH, help="Line width for drawing ArUco markers (default: 5)")
    parser.add_argument("--marker_font_size", type=float, default=FONT_SCALE, help="Font size (scale) for marker IDs (default: 1.5)")
    parser.add_argument("--marker_text_thickness", type=int, default=TEXT_WIDTH, help="Text thickness for marker IDs (default: 3)")
    parser.add_argument("--corner_radius", type=int, default=1, help="Radius for drawing Charuco corners (default: 10)")
    parser.add_argument("--corner_line_width", type=int, default=LINE_WIDTH, help="Line width for drawing Charuco corners (default: 3)")
    parser.add_argument("--corner_font_size", type=float, default=FONT_SCALE, help="Font size (scale) for Charuco corner IDs (default: 1.2)")
    parser.add_argument("--corner_text_thickness", type=int, default=TEXT_WIDTH, help="Text thickness for Charuco corner IDs (default: 2)")
    args = parser.parse_args()

    # 1. Load the Image
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        sys.exit(1)
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at '{image_path}'.")
        sys.exit(1)
        
    h, w = img.shape[:2]
    print(f"Loaded image: '{image_path}'")
    print(f"Image Resolution: {w} x {h} pixels")
    
    # 2. Define the Charuco Board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_length, args.marker_length, dictionary)
    
    # Set up detector parameters with subpixel corner refinement
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    
    # 3. Detect Charuco Board
    print("\n--- Board Detection ---")
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)
    
    num_markers = len(marker_ids) if marker_ids is not None else 0
    num_corners = len(charuco_corners) if charuco_corners is not None else 0
    
    print(f"Detected {num_markers} ArUco markers.")
    print(f"Detected {num_corners} Charuco corners.")
    
    # Print out detected marker corners
    if marker_ids is not None:
        print("\n--- ArUco Marker Coordinates ---")
        for i, mid in enumerate(marker_ids):
            mid_val = mid[0]
            corners = marker_corners[i][0]
            print(f"Marker ID {mid_val:2d}:")
            print(f"  Top-Left:     ({corners[0][0]:7.2f}, {corners[0][1]:7.2f}) px")
            print(f"  Top-Right:    ({corners[1][0]:7.2f}, {corners[1][1]:7.2f}) px")
            print(f"  Bottom-Right: ({corners[2][0]:7.2f}, {corners[2][1]:7.2f}) px")
            print(f"  Bottom-Left:  ({corners[3][0]:7.2f}, {corners[3][1]:7.2f}) px")
            # Calculate marker center
            center = corners.mean(axis=0)
            print(f"  Center:       ({center[0]:7.2f}, {center[1]:7.2f}) px")
            
    # Print out detected Charuco corners
    if charuco_corners is not None:
        print("\n--- Charuco Chessboard Corner Coordinates (Subpixel) ---")
        for i, cid in enumerate(charuco_ids):
            cid_val = cid[0]
            corner = charuco_corners[i][0]
            # Get 3D coordinate corresponding to this corner ID
            pt3d = board.getChessboardCorners()[cid_val]
            print(f"Corner ID {cid_val:2d}: 2D = ({corner[0]:7.2f}, {corner[1]:7.2f}) px | 3D = ({pt3d[0]:4.1f}, {pt3d[1]:4.1f}, {pt3d[2]:4.1f}) mm")

    # 4. Estimate Pixel-to-mm Conversion Factor in the Image Plane
    print("\n--- Pixel-to-mm Conversion Analysis ---")
    h_px_dists = []
    v_px_dists = []
    
    if charuco_corners is not None and len(charuco_corners) > 1:
        # Check all pairs of detected corners to find adjacent ones in the grid
        for i in range(len(charuco_ids)):
            id1 = charuco_ids[i][0]
            r1, c1 = id1 // (args.squares_x - 1), id1 % (args.squares_x - 1)
            p1 = charuco_corners[i][0]
            for j in range(i+1, len(charuco_ids)):
                id2 = charuco_ids[j][0]
                r2, c2 = id2 // (args.squares_x - 1), id2 % (args.squares_x - 1)
                p2 = charuco_corners[j][0]
                
                # Horizontally adjacent in the corner grid
                if r1 == r2 and abs(c1 - c2) == 1:
                    dist_px = np.linalg.norm(p1 - p2)
                    h_px_dists.append(dist_px)
                # Vertically adjacent in the corner grid
                if c1 == c2 and abs(r1 - r2) == 1:
                    dist_px = np.linalg.norm(p1 - p2)
                    v_px_dists.append(dist_px)
                    
        if h_px_dists:
            avg_h_px = np.mean(h_px_dists)
            mm_per_px_h = args.square_length / avg_h_px
            px_per_mm_h = avg_h_px / args.square_length
            print(f"Horizontal grid pitch: {avg_h_px:.2f} px -> {mm_per_px_h:.6f} mm/px ({px_per_mm_h:.2f} px/mm)")
        else:
            mm_per_px_h, avg_h_px = None, None
            print("Horizontal scale could not be computed (no adjacent horizontal corners detected).")
            
        if v_px_dists:
            avg_v_px = np.mean(v_px_dists)
            mm_per_px_v = args.square_length / avg_v_px
            px_per_mm_v = avg_v_px / args.square_length
            print(f"Vertical grid pitch:   {avg_v_px:.2f} px -> {mm_per_px_v:.6f} mm/px ({px_per_mm_v:.2f} px/mm)")
        else:
            mm_per_px_v, avg_v_px = None, None
            print("Vertical scale could not be computed (no adjacent vertical corners detected).")
    else:
        mm_per_px_h, mm_per_px_v = None, None
        print("Not enough Charuco corners detected to estimate local pixel-to-mm conversion factors.")

    # 4.5. Estimate Collinearity Distortion (Line Straightness)
    collin_max, collin_rms = calculate_collinearity_distortion(charuco_corners, charuco_ids, args.squares_x)
    if collin_max is not None:
        print("\n--- Collinearity Distortion Analysis (Line Straightness) ---")
        print(f"  Max line deviation: {collin_max:.3f} pixels")
        print(f"  RMS line deviation: {collin_rms:.3f} pixels")
    else:
        print("\n--- Collinearity Distortion Analysis (Line Straightness) ---")
        print("  Not enough points per row/col to fit straight lines (need >= 3).")

    # 5. Pose Estimation and 3D Rotation Angles
    annotated_img = img.copy()
    pose_depth = None
    mm_per_px_pose = None
    rvec, tvec = None, None
    euler_xyz = None
    reproj_mean = None
    reproj_max = None
    
    if charuco_corners is not None and len(charuco_corners) >= 4:
        # Form 3D-2D point correspondences
        all_obj_points = board.getChessboardCorners()
        obj_points = []
        img_points = []
        for i in range(len(charuco_ids)):
            cid = charuco_ids[i][0]
            obj_points.append(all_obj_points[cid])
            img_points.append(charuco_corners[i][0])
            
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        
        # Estimate Camera Intrinsics
        # Since we do not have a calibration file, we construct a virtual pinhole camera matrix:
        f = args.focal_length if args.focal_length is not None else float(max(w, h))
        cx = w / 2.0
        cy = h / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        
        print("\n--- Camera Pose Estimation & 3D Angles ---")
        print("Assumed Camera Intrinsic Matrix (K):")
        print(f"  Focal Length (f): {f:.1f} pixels")
        print(f"  Optical Center:   ({cx:.1f}, {cy:.1f}) px")
        
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeffs)
        if success:
            # Translation vector (tvec)
            pose_depth = tvec[2][0]
            print(f"\nTranslation Vector (tvec):")
            print(f"  X: {tvec[0][0]:.3f} mm")
            print(f"  Y: {tvec[1][0]:.3f} mm")
            print(f"  Z (Depth): {pose_depth:.3f} mm")
            
            # Pixel-to-mm conversion factor based on pose: mm_per_px = Z / f
            mm_per_px_pose = pose_depth / f
            print(f"Pose-based pixel-to-mm scale (Z/f): {mm_per_px_pose:.6f} mm/px ({1.0/mm_per_px_pose:.2f} px/mm)")
            
            # Rotation Matrix and Euler Angles (Roll, Pitch, Yaw)
            R, _ = cv2.Rodrigues(rvec)
            r_obj = R_scipy.from_matrix(R)
            
            # Standard 'xyz' intrinsic Euler sequence: Rotation around X (Pitch), Y (Yaw), Z (Roll)
            euler_xyz = r_obj.as_euler('xyz', degrees=True)
            pitch_x, yaw_y, roll_z = euler_xyz[0], euler_xyz[1], euler_xyz[2]
            
            print(f"\nRotation Vector (rvec): {rvec.flatten()}")
            print("Board Orientation (Euler angles in 'xyz' sequence):")
            print(f"  Pitch (X-rotation): {pitch_x:+.3f}°")
            print(f"  Yaw   (Y-rotation): {yaw_y:+.3f}°")
            print(f"  Roll  (Z-rotation): {roll_z:+.3f}°")
            
            # Also print ZYX convention for comparison
            euler_zyx = r_obj.as_euler('zyx', degrees=True)
            print(f"Board Orientation (Euler angles in 'zyx' sequence):")
            print(f"  Roll: {euler_zyx[2]:+.3f}°, Pitch: {euler_zyx[1]:+.3f}°, Yaw: {euler_zyx[0]:+.3f}°")
            
            # Calculate Reprojection Error (Ideal Pinhole vs. Actual)
            projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, dist_coeffs)
            reproj_errors = np.linalg.norm(img_points - projected_points.squeeze(), axis=1)
            reproj_mean = float(np.mean(reproj_errors))
            reproj_max = float(np.max(reproj_errors))
            
            print(f"\nReprojection Error (Ideal Pinhole vs Actual Detected):")
            print(f"  Mean Reprojection Error: {reproj_mean:.3f} pixels")
            print(f"  Max Reprojection Error:  {reproj_max:.3f} pixels")

            # 6. Draw 3D axes on the board center
            cv2.drawFrameAxes(annotated_img, K, dist_coeffs, rvec, tvec, length=args.square_length * 2.0, thickness=3)
        else:
            print("Pose estimation (solvePnP) failed.")
    else:
        print("\nPose estimation skipped: not enough Charuco corners detected (need >= 4).")

    # 7. Draw 2D Overlays on the Image
    # A. Draw detected ArUco markers
    if marker_ids is not None:
        draw_detected_markers_custom(
            annotated_img, 
            marker_corners, 
            marker_ids, 
            line_thickness=args.marker_line_width, 
            font_scale=args.marker_font_size, 
            text_thickness=args.marker_text_thickness
        )
        
    # B. Draw detected Charuco corners
    if charuco_corners is not None:
        draw_detected_corners_charuco_custom(
            annotated_img, 
            charuco_corners, 
            charuco_ids, 
            corner_radius=args.corner_radius, 
            line_thickness=args.corner_line_width, 
            font_scale=args.corner_font_size, 
            text_thickness=args.corner_text_thickness
        )

    # C. Overlay Text Information (Roll, Pitch, Yaw, Pixel-to-mm scale)
    y_offset = 40
    line_height = 35
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = FONT_SCALE
    color = (0, 255, 255) # Yellow
    thickness = LINE_WIDTH
    
    cv2.putText(annotated_img, "Charuco Board Analysis", (30, y_offset), font, 1.1, (0, 255, 0), 3)
    y_offset += 45
    
    cv2.putText(annotated_img, f"Resolution: {w}x{h} px", (30, y_offset), font, font_scale, color, thickness)
    y_offset += line_height
    
    cv2.putText(annotated_img, f"Markers: {num_markers} | Corners: {num_corners}", (30, y_offset), font, font_scale, color, thickness)
    y_offset += line_height
    
    if mm_per_px_h is not None and mm_per_px_v is not None:
        cv2.putText(annotated_img, f"Scale (X): {mm_per_px_h:.5f} mm/px ({px_per_mm_h:.1f} px/mm)", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        cv2.putText(annotated_img, f"Scale (Y): {mm_per_px_v:.5f} mm/px ({px_per_mm_v:.1f} px/mm)", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
    if euler_xyz is not None:
        cv2.putText(annotated_img, f"Pose Z (depth): {pose_depth:.1f} mm", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        cv2.putText(annotated_img, f"Roll (Z-rot):  {euler_xyz[2]:+.2f} deg", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        cv2.putText(annotated_img, f"Pitch (X-rot): {euler_xyz[0]:+.2f} deg", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        cv2.putText(annotated_img, f"Yaw (Y-rot):   {euler_xyz[1]:+.2f} deg", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
        
    if collin_max is not None:
        cv2.putText(annotated_img, f"Max Line Distort: {collin_max:.2f} px", (30, y_offset), font, font_scale, color, thickness)
        y_offset += line_height
    if reproj_mean is not None:
        cv2.putText(annotated_img, f"Mean Reproj Error: {reproj_mean:.2f} px", (30, y_offset), font, font_scale, color, thickness)
        
    # 8. Save/Export the annotated image
    input_dir, input_name = os.path.split(image_path)
    base_name, ext = os.path.splitext(input_name)
    output_name = f"{base_name}_detected{ext}"
    
    # We can write it to the same directory or check if there is a 'processed' directory
    if os.path.isdir("processed"):
        output_path = os.path.join("processed", output_name)
    else:
        output_path = os.path.join(input_dir, output_name)
        
    cv2.imwrite(output_path, annotated_img)
    print(f"\nSaved annotated image to: '{output_path}'")
    print("Done!")

if __name__ == "__main__":
    main()
