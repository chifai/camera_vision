#!/usr/bin/env python3
"""
Charuco Undistort & Rectification Script

This script takes a raw image, applies camera matrix K, lens distortion parameters 
(k1, k2, p1, p2, k3) and rotation vector (rvec) obtained from detect_charuco.py 
to generate an undistorted and/or perspective-rectified output image.

Usage:
    python3 undistort_rectify.py <path_to_image> [options]

Example:
    python3 undistort_rectify.py raw/cameraaligner_data/rightcamera_chauro.png --rectify

Author: Antigravity Code Assistant
Date: July 2, 2026
"""

import os
import sys
import argparse
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Apply lens undistortion and rotation rectification to an image.")
    parser.add_argument("image_path", type=str, help="Path to the input raw image.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output image (default: <input_name>_undistorted.png)")
    
    # Distortion parameters (defaults set to the values solved from detect_charuco.py)
    parser.add_argument("--k1", type=float, default=0.0, help="Radial distortion coefficient k1 (default: -6.040867)")
    parser.add_argument("--k2", type=float, default=0.0, help="Radial distortion coefficient k2 (default: 72.86274)")
    parser.add_argument("--p1", type=float, default=0.0, help="Tangential distortion coefficient p1 (default: -0.0005297)")
    parser.add_argument("--p2", type=float, default=0.0, help="Tangential distortion coefficient p2 (default: 0.035767)")
    parser.add_argument("--k3", type=float, default=0.0, help="Radial distortion coefficient k3 (default: -250.6142)")
    
    # Rotational parameters (default set to rotation solved from detect_charuco.py)
    parser.add_argument("--rvec", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Rotation vector (rvec) from pose estimation (default: 0.03908682 -0.13819391 0.04552815)")
    
    # Rectification control
    parser.add_argument("--rectify", action="store_true", 
                        help="Enable perspective rectification (removes roll/pitch/yaw tilt of the board, flattening it).")
    parser.add_argument("--focal_length", type=float, default=None, 
                        help="Assumed camera focal length in pixels. Defaults to max(width, height)")

    args = parser.parse_args()

    # 1. Load the Image
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image '{image_path}'.")
        sys.exit(1)
        
    h, w = img.shape[:2]
    print(f"Loaded image: '{image_path}' ({w} x {h} px)")

    # 2. Define Camera Matrix (K) and Distortion Vector (D)
    f = args.focal_length if args.focal_length is not None else float(max(w, h))
    cx = w / 2.0
    cy = h / 2.0
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    D = np.array([args.k1, args.k2, args.p1, args.p2, args.k3], dtype=np.float64)
    
    print("\n--- Calibration Inputs ---")
    print("Camera Matrix K:")
    print(K)
    print(f"Distortion Vector D (k1, k2, p1, p2, k3):")
    print(f"  [{args.k1:+.6e}, {args.k2:+.6e}, {args.p1:+.6e}, {args.p2:+.6e}, {args.k3:+.6e}]")

    # 3. Compute Undistortion and Rectification Maps
    if args.rectify:
        # Convert rotation vector to 3x3 rotation matrix R
        rvec = np.array(args.rvec, dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        print("\nApplying Perspective Rectification (Rotation correction):")
        print(f"Rotation vector (rvec): {rvec}")
        print("Rotation Matrix (R):")
        print(R)
        
        # initUndistortRectifyMap will correct for both lens distortion (D) and board rotation (R).
        # We pass R to align the virtual camera with the plane of the board.
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, K, (w, h), cv2.CV_32FC1)
        mode_str = "undistorted_rectified"
    else:
        print("\nApplying standard lens undistortion (no rotation correction):")
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
        mode_str = "undistorted"

    # 4. Warp the Image
    output_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    # 5. Save Output
    if args.output_path is None:
        input_dir, input_name = os.path.split(image_path)
        base_name, ext = os.path.splitext(input_name)
        output_name = f"{base_name}_{mode_str}{ext}"
        
        if os.path.isdir("processed"):
            output_path = os.path.join("processed", output_name)
        else:
            output_path = os.path.join(input_dir, output_name)
    else:
        output_path = args.output_path

    cv2.imwrite(output_path, output_img)
    print(f"\nSuccessfully generated and saved output to: '{output_path}'")
    print("Done!")

if __name__ == "__main__":
    main()
