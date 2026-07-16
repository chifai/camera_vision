#!/usr/bin/env python3
"""
Rectify and Undistort Folder Script

This script reads a camera calibration JSON file (calibration_results.json)
from a folder, processes all images in that folder to:
1. Estimate the 3D board pose (rotation/translation vectors).
2. Rectify and undistort the image.
3. Perform a second-pass ChArUco detection on the rectified image.
4. Overlay ChArUco corners, adjacent distances (in mm), and metadata (RPY and scale).
5. Output the annotated results to a subfolder.

Usage:
    python3 rectify_folder.py <folder_path> [options]
"""

import os
import sys
import json
import argparse
import tempfile
import cv2
import numpy as np
from detect_charuco import CameraCalibration, DICT_MAPPING

def main():
    parser = argparse.ArgumentParser(description="Batch rectify and annotate images in a folder using calibration results.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images and calibration_results.json.")
    parser.add_argument("--squares_x", type=int, default=7, help="Number of board squares along X (default: 7)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of board squares along Y (default: 5)")
    parser.add_argument("--square_length", type=float, default=3.5, help="Square length in mm (default: 3.5)")
    parser.add_argument("--marker_length", type=float, default=2.0, help="Marker length in mm (default: 2.0)")
    parser.add_argument("--dict_id", type=str, default="DICT_4X4_250", choices=list(DICT_MAPPING.keys()),
                        help="ArUco dictionary name (default: DICT_4X4_250)")
    parser.add_argument("--crop", action="store_true", default=True,
                        help="Crop out black boundaries after undistortion (default: True)")
    parser.add_argument("--no_crop", action="store_false", dest="crop",
                        help="Do not crop out black boundaries")
    parser.add_argument("--auto_mirror", action="store_true", default=True,
                        help="Automatically try mirrored images (default: True)")
    parser.add_argument("--no_auto_mirror", action="store_false", dest="auto_mirror",
                        help="Disable auto mirroring")
    parser.add_argument("--subfolder_name", type=str, default="rectified_processed",
                        help="Name of the output subfolder (default: rectified_processed)")
    parser.add_argument("--corner_refinement_method", type=str, default="apriltag",
                        choices=["none", "subpix", "contour", "apriltag"],
                        help="Subpixel corner refinement method (default: apriltag)")
    
    args = parser.parse_args()
    
    # 1. Resolve calibration_results.json
    calib_json_path = os.path.join(args.folder_path, "calibration_results.json")
    if not os.path.exists(calib_json_path):
        print(f"Error: Calibration file not found at: '{calib_json_path}'", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(calib_json_path, "r") as f:
            calib_data = json.load(f)
        K = np.array(calib_data["K"], dtype=np.float64)
        dist_coeffs = np.array(calib_data["dist_coeffs"], dtype=np.float64)
    except Exception as e:
        print(f"Error reading calibration JSON: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Loaded camera calibration successfully.")
    print("Camera Matrix K:")
    print(K)
    print("Distortion Coefficients:")
    print(dist_coeffs)
    
    # 2. Get list of valid images
    valid_extensions = {".png", ".jpg", ".jpeg"}
    image_files = []
    for f in os.listdir(args.folder_path):
        if os.path.splitext(f)[1].lower() in valid_extensions:
            # Skip any previously outputted processed images (e.g. undistorted, corners_plot)
            if "_undistorted" in f or "_corners_plot" in f or "_rectified" in f or "_comparison" in f:
                continue
            image_files.append(os.path.join(args.folder_path, f))
            
    image_files = sorted(image_files)
    print(f"\nFound {len(image_files)} raw images to process.")
    if not image_files:
        print("No raw images found in the folder to process.")
        sys.exit(0)
        
    # 3. Create output subfolder
    output_subfolder = os.path.join(args.folder_path, args.subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    print(f"Annotated rectified images will be saved to: '{output_subfolder}'")
    
    dictionary_id = DICT_MAPPING[args.dict_id]
    
    # 4. Process each image
    for filepath in image_files:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        ext = os.path.splitext(filepath)[1]
        print(f"\nProcessing '{os.path.basename(filepath)}'...")
        
        try:
            calib = CameraCalibration(
                image_path=filepath,
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_length=args.square_length,
                marker_length=args.marker_length,
                dictionary_id=dictionary_id,
                corner_refinement_method=args.corner_refinement_method,
                try_refine_markers=True
            )
            # Override K and dist_coeffs with calibration results
            calib.K = K.copy()
            calib.dist_coeffs = dist_coeffs.copy()
            
            # Detect to get rotation vector rvec
            success = calib.detect_charuco(auto_mirror=args.auto_mirror)
            if not success or calib.rvec is None:
                print(f"  Skipping: ChArUco detection or pose estimation failed on raw image.")
                continue
                
            # Save original annotated image (before rectification)
            orig_annotated_path = os.path.join(output_subfolder, f"{base_name}_annotated{ext}")
            cv2.imwrite(orig_annotated_path, calib.get_annotated_image())
            print(f"  Saved original annotated image: '{orig_annotated_path}'")
                
            # Perform rectification and undistortion
            rectified_img = calib.undistort(
                rectify=True,
                rvec=calib.rvec,
                dist_coeffs=calib.dist_coeffs,
                crop=args.crop
            )
            
            # Save rectified image to temporary file for second-pass ChArUco detection
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_rectified_path = os.path.join(tmpdir, "rectified_tmp.png")
                cv2.imwrite(tmp_rectified_path, rectified_img)
                
                # Second-pass detection on the rectified image
                rectified_calib = CameraCalibration(
                    image_path=tmp_rectified_path,
                    squares_x=args.squares_x,
                    squares_y=args.squares_y,
                    square_length=args.square_length,
                    marker_length=args.marker_length,
                    dictionary_id=dictionary_id,
                    corner_refinement_method=args.corner_refinement_method,
                    try_refine_markers=True
                )
                
                # We assume rectified image has no lens distortion and is identity-rectified
                rectified_calib.K = K.copy()
                rectified_calib.dist_coeffs = np.zeros(5, dtype=np.float64)
                
                rect_success = rectified_calib.detect_charuco(auto_mirror=args.auto_mirror)
                if not rect_success:
                    print(f"  Warning: Corner detection failed on rectified image.")
                    # Fallback: save raw rectified image without overlays
                    out_path = os.path.join(output_subfolder, f"{base_name}_rectified{ext}")
                    cv2.imwrite(out_path, rectified_img)
                    print(f"  Saved raw rectified image: '{out_path}'")
                    continue
                
                print(f"  Detected {len(rectified_calib.charuco_corners)} corners on rectified image.")
                
                # Overlay corners & adjacent distances in mm
                annotated = rectified_calib.get_corners_distance_plot()
                
                # Overlay RPY and scale metadata (pose from original calibration/detection)
                h_img, w_img = annotated.shape[:2]
                num_markers = len(rectified_calib.marker_ids) if rectified_calib.marker_ids is not None else 0
                num_corners = len(rectified_calib.charuco_corners) if rectified_calib.charuco_corners is not None else 0
                
                CameraCalibration.overlay_text_info(
                    annotated, w_img, h_img, num_markers, num_corners,
                    mm_per_px_h=rectified_calib.mm_per_px_h,
                    mm_per_px_v=rectified_calib.mm_per_px_v,
                    px_per_mm_h=rectified_calib.px_per_mm_h,
                    px_per_mm_v=rectified_calib.px_per_mm_v,
                    euler_xyz=rectified_calib.euler_xyz,
                    pose_depth=rectified_calib.pose_depth,
                    font_scale=0.75,
                    thickness=2
                )
                
                # Save raw unannotated rectified image
                raw_rectified_path = os.path.join(output_subfolder, f"{base_name}_rectified{ext}")
                cv2.imwrite(raw_rectified_path, rectified_img)
                print(f"  Saved raw rectified image: '{raw_rectified_path}'")
                
                # Save annotated rectified image
                out_path = os.path.join(output_subfolder, f"{base_name}_rectified_annotated{ext}")
                cv2.imwrite(out_path, annotated)
                print(f"  Saved annotated rectified image: '{out_path}'")
                
                # Save JSON results and CSV corners for this rectified run as well
                json_path = os.path.join(output_subfolder, f"{base_name}_results.json")
                rectified_calib.save_results_json(
                    json_path,
                    image_path=out_path,
                    squares_x=args.squares_x,
                    squares_y=args.squares_y,
                    square_length=args.square_length,
                    marker_length=args.marker_length
                )
                
                csv_path = os.path.join(output_subfolder, f"{base_name}_corners.csv")
                rectified_calib.save_corners_csv(csv_path)
                
        except Exception as e:
            print(f"  Error processing image: {e}", file=sys.stderr)
            
    print("\nBatch folder processing completed successfully!")

if __name__ == "__main__":
    main()
