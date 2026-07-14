import os
import sys
import time
import argparse
import urllib.request
import urllib.error
import numpy as np
import cv2

# Add current directory to path so we can import detect_charuco
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from detect_charuco import CameraCalibration

DICT_MAPPING = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
    'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
    'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
    'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
    'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
    'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
    'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
    'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
    'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
    'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000
}

def main():
    parser = argparse.ArgumentParser(description="Acquire calibration images from an HTTP endpoint and perform camera calibration.")
    parser.add_argument("url", type=str, help="HTTP URL to fetch images from")
    parser.add_argument("-n", "--num_images", type=int, default=20, help="Number of valid images to collect (default: 10)")
    parser.add_argument("-o", "--folder", type=str, default="calib_images", help="Folder to save valid images (default: calib_images)")
    parser.add_argument("--squares_x", type=int, default=7, help="Number of squares along X-axis (default: 7)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of squares along Y-axis (default: 5)")
    parser.add_argument("--square_length", type=float, default=3.5, help="Square length in mm (default: 3.5)")
    parser.add_argument("--marker_length", type=float, default=2.0, help="Marker length in mm (default: 2.0)")
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_250", choices=list(DICT_MAPPING.keys()), help="Dictionary name (default: DICT_4X4_250)")
    parser.add_argument("--interval", type=float, default=2.0, help="Request interval in seconds (default: 1.0)")
    parser.add_argument("--auto_mirror", action="store_true", default=True, help="Automatically try mirrored image to maximize corner detection (default: True)")
    parser.add_argument("--no_auto_mirror", action="store_false", dest="auto_mirror", help="Disable auto mirroring")
    
    args = parser.parse_args()
    
    # Map dictionary name to constant
    dict_id = DICT_MAPPING[args.dictionary]
    
    # Initialize ChArUco detector parameters
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_length, args.marker_length, dictionary)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board, detectorParams=detector_params)
    
    # Create output directory
    os.makedirs(args.folder, exist_ok=True)
    
    print(f"Starting acquisition from: {args.url}")
    print(f"Targeting {args.num_images} valid images with >= 4 ChArUco corners.")
    print(f"Saving images to folder: {args.folder}")
    print(f"Press Ctrl+C to terminate early.\n")
    
    valid_count = 0
    attempt_count = 0
    
    try:
        while valid_count < args.num_images:
            attempt_count += 1
            start_time = time.time()
            
            try:
                # Fetch image from URL
                req = urllib.request.Request(args.url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    img_bytes = response.read()
                
                # Decode image
                img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"[{attempt_count}] Error: Failed to decode image from response.")
                else:
                    # Detect board (try original first)
                    corners, ids, _, _ = detector.detectBoard(img)
                    num_corners = len(corners) if corners is not None else 0
                    is_flipped = False
                    
                    if args.auto_mirror:
                        flipped_img = cv2.flip(img, 1)
                        flipped_corners, flipped_ids, _, _ = detector.detectBoard(flipped_img)
                        n_flipped_corners = len(flipped_corners) if flipped_corners is not None else 0
                        if n_flipped_corners > num_corners:
                            corners = flipped_corners
                            ids = flipped_ids
                            num_corners = n_flipped_corners
                            img = flipped_img
                            is_flipped = True
                    
                    mirror_suffix = " (Mirrored)" if is_flipped else ""
                    print(f"[{attempt_count}] Fetched image{mirror_suffix}. Corners found: {num_corners}")
                    
                    if num_corners >= 4:
                        save_path = os.path.join(args.folder, f"{valid_count}.png")
                        cv2.imwrite(save_path, img)
                        valid_count += 1
                        print(f"  -> Valid! Saved to {save_path} ({valid_count}/{args.num_images})")
                    else:
                        print("  -> Invalid: Not enough corners (needs >= 4).")
                        
            except urllib.error.URLError as e:
                print(f"[{attempt_count}] HTTP/Network Error: {e.reason}")
            except Exception as e:
                print(f"[{attempt_count}] Unexpected Error: {e}")
                
            # Sleep to maintain 1-second interval
            elapsed = time.time() - start_time
            sleep_time = max(0.0, args.interval - elapsed)
            if valid_count < args.num_images:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nAcquisition interrupted by user.")
        
    print(f"\nAcquisition phase complete. Collected {valid_count} valid images.")
    
    if valid_count < 3:
        print("Error: Need at least 3 valid images to perform multi-image calibration.")
        sys.exit(1)
        
    # Start camera calibration using detect_charuco functions
    print("\n===========================================================")
    print(" Running Camera Calibration using detect_charuco...")
    print("===========================================================")
    
    success, results = CameraCalibration.calibrate_from_directory(
        directory_path=args.folder,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_id=dict_id
    )
    
    if success:
        print("\n===========================================================")
        print(" MULTI-IMAGE CALIBRATION RESULTS")
        print("===========================================================")
        print(f"Successfully Calibrated Views: {len(results['image_paths'])}")
        print(f"Reprojection Error (RMS):       {results['reproj_error']:.4f} pixels")
        print("\nSolved Camera Matrix K:")
        print(results['K'])
        print("\nSolved Distortion Coefficients D:")
        print(f"  k1, k2, p1, p2, k3 = {results['dist_coeffs'].flatten().tolist()}")
        
        # Save results to a json file in the directory
        import json
        out_file = os.path.join(args.folder, "calibration_results.json")
        data = {
            'reproj_error': float(results['reproj_error']),
            'K': results['K'].tolist(),
            'dist_coeffs': results['dist_coeffs'].flatten().tolist(),
            'image_size': results['image_size'],
            'image_paths': [os.path.basename(p) for p in results['image_paths']]
        }
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nSaved calibration results to: '{out_file}'")
        
        # Run individual analysis to output scale metrics (mm/pixel)
        print("\n--- Scale Analysis per Image View ---")
        for filepath in results['image_paths']:
            try:
                calib = CameraCalibration(
                    image_path=filepath,
                    squares_x=args.squares_x,
                    squares_y=args.squares_y,
                    square_length=args.square_length,
                    marker_length=args.marker_length,
                    dictionary_id=dict_id
                )
                if calib.detect_charuco():
                    print(f"\nImage: {os.path.basename(filepath)}")
                    if calib.mm_per_px_h is not None:
                        print(f"  Horizontal scale: {calib.mm_per_px_h:.6f} mm/px ({calib.px_per_mm_h:.2f} px/mm)")
                        print(f"  Vertical scale:   {calib.mm_per_px_v:.6f} mm/px ({calib.px_per_mm_v:.2f} px/mm)")
                    if calib.pose_depth is not None:
                        print(f"  Estimated Depth:  {calib.pose_depth:.2f} mm")
                        print(f"  Scale from pose:  {calib.mm_per_px_pose:.6f} mm/px")
            except Exception as e:
                print(f"Warning: Failed to run analysis on {os.path.basename(filepath)}: {e}")
        print("===========================================================")
    else:
        print("\nCalibration failed.")
        print("===========================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
