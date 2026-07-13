#!/usr/bin/env python3
"""
Charuco Undistort & Rectification Script

This script takes a raw image, applies camera matrix K, lens distortion parameters 
(k1, k2, p1, p2, k3) and rotation vector (rvec) obtained from detect_charuco.py 
to generate an undistorted and/or perspective-rectified output image, as well as
a side-by-side comparison image.

Usage:
    python3 undistort_rectify.py <path_to_image> [options]

Example:
    python3 undistort_rectify.py raw/cameraaligner_data/rightcamera_chauro.png --rectify

Author: Antigravity Code Assistant
Date: July 3, 2026
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Try importing CameraCalibration for drawing annotations in comparison mode
try:
    from detect_charuco import CameraCalibration
    HAS_CALIB_CLASS = True
except ImportError:
    HAS_CALIB_CLASS = False

def main():
    parser = argparse.ArgumentParser(description="Apply lens undistortion and rotation rectification to an image.")
    parser.add_argument("image_path", type=str, help="Path to the input raw image.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output image (default: <input_name>_undistorted.png)")
    
    # Distortion parameters (defaults set to the values solved from detect_charuco.py)
    parser.add_argument("--k1", type=float, default=0.0, help="Radial distortion coefficient k1 (default: 0.0)")
    parser.add_argument("--k2", type=float, default=0.0, help="Radial distortion coefficient k2 (default: 0.0)")
    parser.add_argument("--p1", type=float, default=0.0, help="Tangential distortion coefficient p1 (default: 0.0)")
    parser.add_argument("--p2", type=float, default=0.0, help="Tangential distortion coefficient p2 (default: 0.0)")
    parser.add_argument("--k3", type=float, default=0.0, help="Radial distortion coefficient k3 (default: 0.0)")
    
    # Rotational parameters (default set to rotation solved from detect_charuco.py)
    parser.add_argument("--rvec", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Rotation vector (rvec) from pose estimation (default: 0.0, 0.0, 0.0)")
    
    # Rectification and Comparison controls
    parser.add_argument("--rectify", action="store_true", 
                        help="Enable perspective rectification (removes roll/pitch/yaw tilt of the board, flattening it).")
    parser.add_argument("--crop", action="store_true",
                        help="Crop the output image to remove invalid boundary/edge pixels (uses optimal camera matrix calculation).")
    parser.add_argument("--focal_length", type=float, default=None, 
                        help="Assumed camera focal length in pixels. Defaults to max(width, height)")
    parser.add_argument("--no_comparison", action="store_true",
                        help="Disable generating the side-by-side comparison image.")

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
    K_new = K.copy()
    roi = None
    if args.crop:
        # Calculate optimal camera matrix to crop out black out-of-bounds pixels
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)

    if args.rectify:
        # Convert rotation vector to 3x3 rotation matrix R
        rvec = np.array(args.rvec, dtype=np.float64)
        R_mat, _ = cv2.Rodrigues(rvec)
        # Use transpose (inverse) for rectification
        R_rect = R_mat.T
        print("\nApplying Perspective Rectification (Rotation correction):")
        print(f"Rotation vector (rvec): {rvec}")
        print("Rectification Rotation Matrix (R.T):")
        print(R_rect)
        
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R_rect, K_new, (w, h), cv2.CV_32FC1)
        mode_str = "undistorted_rectified"
    else:
        print("\nApplying standard lens undistortion (no rotation correction):")
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K_new, (w, h), cv2.CV_32FC1)
        mode_str = "undistorted"

    # 4. Warp the Image
    output_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Apply crop if enabled
    if args.crop and roi is not None:
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            output_img = output_img[y:y+h_roi, x:x+w_roi]
        mode_str += "_cropped"

    # 5. Save Main Output
    if args.output_path is None:
        input_dir, input_name = os.path.split(image_path)
        base_name, ext = os.path.splitext(input_name)
        output_name = f"{base_name}_{mode_str}{ext}"
        
        out_dir = "processed" if os.path.isdir("processed") else input_dir
        output_path = os.path.join(out_dir, output_name)
    else:
        output_path = args.output_path
        input_dir, input_name = os.path.split(image_path)
        base_name, ext = os.path.splitext(input_name)
        out_dir = os.path.split(output_path)[0]

    cv2.imwrite(output_path, output_img)
    print(f"\nSuccessfully generated and saved output to: '{output_path}'")

    # 6. Generate and Save Side-by-Side Comparison Image (if not disabled)
    if not args.no_comparison:
        annotated_img = None
        title_left = "Original Raw View"
        
        # Try running Charuco detection to annotate the original view
        if HAS_CALIB_CLASS:
            try:
                calib = CameraCalibration(image_path)
                success = calib.detect_charuco(focal_length=f)
                if success:
                    # Retrieve the annotated view from class
                    annotated_img = calib.get_annotated_image()
                    title_left = "Detected & Annotated View"
            except Exception as e:
                print(f"Note: Could not run Charuco board detection for comparison annotations: {e}")
                
        if annotated_img is None:
            annotated_img = img.copy()

        # Helper to add centered title text on a black bar on top of the images
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

        # If the output image was cropped, resize it to match the height of annotated_img
        # so they can be stacked side-by-side using np.hstack
        if output_img.shape[0] != h:
            scale = h / output_img.shape[0]
            new_w = int(output_img.shape[1] * scale)
            output_img_resized = cv2.resize(output_img, (new_w, h), interpolation=cv2.INTER_AREA)
        else:
            output_img_resized = output_img

        # Add titles and stack horizontally
        annotated_titled = add_title(annotated_img, title_left)
        processed_titled = add_title(output_img_resized, "Undistorted & Rectified View" if args.rectify else "Undistorted View")
        
        comparison_img = np.hstack((annotated_titled, processed_titled))
        
        # Save comparison image
        path_comparison = os.path.join(out_dir, f"{base_name}_comparison{'_cropped' if args.crop else ''}{ext}")
        cv2.imwrite(path_comparison, comparison_img)
        print(f"Successfully generated and saved comparison to: '{path_comparison}'")

    print("Done!")

if __name__ == "__main__":
    main()
