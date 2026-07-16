#!/usr/bin/env python3
"""
Batch Undistort & Rectify Images with Corner Detection Overlays

This script loads solved camera calibration parameters (K, distortion coefficients,
and image-specific rotation vectors) from a directory's `calibration_results.json`
file and applies distortion correction + perspective rectification to all raw images.

For each undistorted image, it runs ChArUco corner detection again and generates
an annotated output with:
  - Detected corner dots, connection lines, and mm distances between adjacent corners
  - Row X-mm/pixel and column Y-mm/pixel scale ratios
  - RPY angles and mm/pixel metrics from the calibration pose
"""

import os
import sys
import json
import argparse
import tempfile
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from detect_charuco import CameraCalibration


def batch_undistort(folder_path, args):
    # 1. Load calibration_results.json
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

        # Build rvec lookup map: image basename -> rvec array
        rvec_lookup = {}
        for path, rvec in zip(image_paths, rvecs):
            rvec_lookup[os.path.basename(path)] = np.array(rvec, dtype=np.float64)
        print(f"Mapped rotation vectors for {len(rvec_lookup)} views.")

    except Exception as e:
        print(f"Error parsing calibration JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Find all raw images to process
    valid_exts = {'.png', '.jpg', '.jpeg'}
    images_to_process = []
    for f in os.listdir(folder_path):
        name, ext = os.path.splitext(f)
        if ext.lower() in valid_exts and not name.endswith("_undistorted"):
            images_to_process.append(f)

    if not images_to_process:
        print("No valid raw images found to process.")
        return

    print(f"Found {len(images_to_process)} raw images to process.")

    crop_enabled = not args.no_crop
    print(f"Cropping boundary pixels (crop=True): {crop_enabled}")

    for f in sorted(images_to_process):
        img_path = os.path.join(folder_path, f)
        base_name, ext = os.path.splitext(f)
        rvec = rvec_lookup.get(f)
        rectify_enabled = rvec is not None

        # 3. Undistort (and optionally rectify) using CameraCalibration
        try:
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
            print(f"  Warning: Failed to load '{f}': {e}. Skipping.")
            continue

        undistorted = calib.undistort(rectify=rectify_enabled, rvec=rvec, dist_coeffs=dist_coeffs, crop=crop_enabled)

        # 4. Save standalone undistorted image
        out_name = f"{base_name}_undistorted{ext}"
        out_path = os.path.join(folder_path, out_name)
        cv2.imwrite(out_path, undistorted)
        mode = "undistorted & rectified" if rectify_enabled else "undistorted"
        print(f"  Saved {mode} image: '{out_name}'")

        # 5. Run ChArUco detection on the LENS-UNDISTORTED image (no perspective warp)
        #    Rectification distorts markers too aggressively for reliable re-detection.
        undistorted_only = calib.undistort(rectify=False, dist_coeffs=dist_coeffs, crop=False)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            cv2.imwrite(tmp_path, undistorted_only)

            calib2 = CameraCalibration(
                image_path=tmp_path,
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_length=args.square_length,
                marker_length=args.marker_length,
                corner_refinement_method=args.corner_refinement_method,
                try_refine_markers=args.try_refine_markers
            )
            # Inject calibrated intrinsics; dist is already removed so use zeros
            calib2.K = K
            calib2.dist_coeffs = np.zeros(5, dtype=np.float64)

            detection_success = calib2.detect_charuco()
        finally:
            os.unlink(tmp_path)

        # 6. Build annotated corners-plot on the undistorted image
        if detection_success:
            # get_corners_distance_plot() draws corners, lines, mm distances and row/col ratios
            annotated = calib2.get_corners_distance_plot()

            # Overlay RPY and mm/pixel using CameraCalibration.overlay_text_info
            if rvec is not None:
                h_img, w_img = annotated.shape[:2]
                num_markers = len(calib2.marker_ids) if calib2.marker_ids is not None else 0
                num_corners = len(calib2.charuco_corners) if calib2.charuco_corners is not None else 0
                CameraCalibration.overlay_text_info(
                    annotated, w_img, h_img, num_markers, num_corners,
                    mm_per_px_h=calib2.mm_per_px_h,
                    mm_per_px_v=calib2.mm_per_px_v,
                    px_per_mm_h=calib2.px_per_mm_h,
                    px_per_mm_v=calib2.px_per_mm_v,
                    euler_xyz=calib2.euler_xyz,
                    pose_depth=calib2.pose_depth,
                    font_scale=0.75,
                    thickness=2
                )

            plot_name = f"{base_name}_undistorted_corners_plot{ext}"
            plot_path = os.path.join(folder_path, plot_name)
            cv2.imwrite(plot_path, annotated)
            n_corners = len(calib2.charuco_corners) if calib2.charuco_corners is not None else 0
            print(f"  Saved corners plot ({n_corners} corners): '{plot_name}'")
        else:
            print(f"  Corner detection failed on undistorted image; no corners plot saved.")

    print("\nBatch undistortion, rectification and corner annotation completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Batch undistort/rectify images and overlay corner detection with RPY and mm/pixel metrics"
    )
    parser.add_argument("folder_path", type=str, help="Path to folder containing calibration_results.json and raw images")
    parser.add_argument("--squares_x", type=int, default=7, help="Board squares in X (default: 7)")
    parser.add_argument("--squares_y", type=int, default=5, help="Board squares in Y (default: 5)")
    parser.add_argument("--square_length", type=float, default=3.5, help="Square length in mm (default: 3.5)")
    parser.add_argument("--marker_length", type=float, default=2.0, help="Marker length in mm (default: 2.0)")
    parser.add_argument("--no_crop", action="store_true", help="Disable auto-cropping of black boundary regions")
    parser.add_argument("--corner_refinement_method", type=str, default="apriltag",
                        choices=["none", "subpix", "contour", "apriltag"],
                        help="Corner refinement method for second-pass detection (default: apriltag)")
    parser.add_argument("--try_refine_markers", action="store_true", default=True,
                        help="Try to refine markers in second-pass detection (default: True)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder path '{args.folder_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    batch_undistort(args.folder_path, args)


if __name__ == "__main__":
    main()
