#!/usr/bin/env python3
"""
Batch Undistort & Rectify Images (Standalone & Cropped)

This script loads solved camera calibration parameters (K, distortion coefficients,
and image-specific rotation vectors) from a directory's `calibration_results.json` 
file and applies distortion correction and perspective rectification to all raw 
images in that folder using the CameraCalibration.undistort class method.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from detect_charuco import CameraCalibration

def batch_undistort(folder_path, args):
    # 1. Look for calibration_results.json
    json_path = os.path.join(folder_path, "calibration_results.json")
    if not os.path.exists(json_path):
        print(f"Error: Could not find 'calibration_results.json' in '{folder_path}'", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(json_path, 'r') as f:
            calib_data = json.load(f)
        
        K = np.array(calib_data['K'], dtype=np.float64)
        dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=np.float64)
        image_paths = calib_data.get('image_paths', [])
        rvecs = calib_data.get('rvecs', [])
        
        print("Loaded calibration parameters successfully.")
        print(f"Camera Matrix K:\n{K}")
        print(f"Distortion Coefficients:\n{dist_coeffs}")
        
        # Build rvec lookup map matching image basenames to their rotation vectors
        rvec_lookup = {}
        for path, rvec in zip(image_paths, rvecs):
            rvec_lookup[os.path.basename(path)] = np.array(rvec, dtype=np.float64)
        print(f"Mapped rotation vectors for {len(rvec_lookup)} views.")
        
    except Exception as e:
        print(f"Error parsing calibration JSON file: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Find all images under the folder path
    valid_exts = {'.png', '.jpg', '.jpeg'}
    all_files = os.listdir(folder_path)
    
    images_to_process = []
    for f in all_files:
        name, ext = os.path.splitext(f)
        # Exclude comparison/undistorted images we generate to avoid infinite loops/double processing
        if ext.lower() in valid_exts and not name.endswith("_undistorted"):
            images_to_process.append(f)
            
    if not images_to_process:
        print("No valid raw images found to process.")
        return
        
    print(f"Found {len(images_to_process)} raw images to process.")
    
    # 3. Apply undistort and optional rectification to all images
    crop_enabled = not args.no_crop
    print(f"Cropping boundary pixels (crop=True): {crop_enabled}")
    
    for f in sorted(images_to_process):
        img_path = os.path.join(folder_path, f)
        
        try:
            # Instantiate CameraCalibration for the image
            calib = CameraCalibration(
                image_path=img_path,
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_length=args.square_length,
                marker_length=args.marker_length
            )
            calib.K = K
            calib.dist_coeffs = dist_coeffs
        except Exception as e:
            print(f"  Warning: Failed to load image '{f}' in CameraCalibration: {e}. Skipping.")
            continue
            
        # Get matching rotation vector for perspective rectification
        rvec = rvec_lookup.get(f)
        rectify_enabled = rvec is not None
        
        # Re-use CameraCalibration.undistort method
        undistorted = calib.undistort(rectify=rectify_enabled, rvec=rvec, dist_coeffs=dist_coeffs, crop=crop_enabled)
        
        # Construct output file name
        base_name, ext = os.path.splitext(f)
        out_name = f"{base_name}_undistorted{ext}"
        out_path = os.path.join(folder_path, out_name)
        
        cv2.imwrite(out_path, undistorted)
        if rectify_enabled:
            print(f"  Saved standalone undistorted & rectified image: '{out_name}'")
        else:
            print(f"  Saved standalone undistorted image (no pose for rectification): '{out_name}'")
        
    print("\nBatch undistortion and rectification completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Batch undistort and rectify images from calibration results")
    parser.add_argument("folder_path", type=str, help="Path to folder containing calibration_results.json and raw images")
    parser.add_argument("--squares_x", type=int, default=7, help="Number of board squares in X (default: 7)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of board squares in Y (default: 5)")
    parser.add_argument("--square_length", type=float, default=3.5, help="Square length in mm (default: 3.5)")
    parser.add_argument("--marker_length", type=float, default=2.0, help="Marker length in mm (default: 2.0)")
    parser.add_argument("--no_crop", action="store_true", help="Disable auto-cropping of black boundary regions (crop=False)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder path '{args.folder_path}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)
        
    batch_undistort(args.folder_path, args)

if __name__ == "__main__":
    main()
