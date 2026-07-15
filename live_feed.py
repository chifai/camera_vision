#!/usr/bin/env python3
"""
Live USB Camera Feed & Interactive ChArUco Board Detector

This script opens a USB camera feed, displays it in real-time, and provides
interactive controls to toggle ChArUco corner detection overlays and capture 
snapshots for calibration.

Controls:
    'd' - Toggle real-time ChArUco detection overlay
    's' - Save current frame to the 'raw/' folder
    'q' / ESC - Quit the application
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
from detect_charuco import CameraCalibration

def main():
    parser = argparse.ArgumentParser(description="Live USB Camera Feed & Interactive ChArUco Detector")
    parser.add_argument("--index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=2592, help="Requested frame width (default: 2592)")
    parser.add_argument("--height", type=int, default=1944, help="Requested frame height (default: 1944)")
    
    # Board detection configuration
    parser.add_argument("--squares_x", type=int, default=7, help="Number of board squares in X (default: 7)")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of board squares in Y (default: 5)")
    parser.add_argument("--square_length", type=float, default=3.5, help="Square length in mm (default: 3.5)")
    parser.add_argument("--marker_length", type=float, default=2.0, help="Marker length in mm (default: 2.0)")
    parser.add_argument("--corner_refinement_method", type=str, default="subpix", 
                        choices=["none", "subpix", "contour", "apriltag"],
                        help="Subpixel corner refinement method (default: subpix)")
    parser.add_argument("--try_refine_markers", action="store_true", default=True,
                        help="Try to refine/find missing markers using board layout (default: True)")
    
    args = parser.parse_args()
    
    # 1. Ensure output directory exists
    save_dir = "raw"
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Setup ChArUco board and detector for detection overlay
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    # OpenCV CharucoBoard expects dimensions in meters
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), 
        args.square_length / 1000.0, 
        args.marker_length / 1000.0, 
        dictionary
    )
    detector, _, _ = CameraCalibration.create_detector(
        board,
        corner_refinement_method=args.corner_refinement_method,
        try_refine_markers=args.try_refine_markers
    )
    
    # 3. Open Video Capture
    print(f"Opening USB camera at index {args.index}...")
    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.index}.", file=sys.stderr)
        print("Please check connection and verify device index.", file=sys.stderr)
        sys.exit(1)
        
    # Configure resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Read actual configured settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened successfully! Resolution: {actual_w}x{actual_h} px")
    
    # Interactive variables
    detect_mode = True
    prev_time = time.time()
    
    print("\n-----------------------------------------------------------")
    print(" Interactive Controls:")
    print("   [d] - Toggle ChArUco detection overlay")
    print("   [s] - Capture/Save current frame as image")
    print("   [q] / [ESC] - Exit")
    print("-----------------------------------------------------------")
    
    window_name = f"USB Camera Live Feed (Index {args.index})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to retrieve frame from camera.", file=sys.stderr)
                break
                
            display_frame = frame.copy()
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Run ChArUco detection overlay if enabled
            if detect_mode:
                corners, ids, marker_corners, marker_ids = detector.detectBoard(display_frame)
                
                # Draw marker borders
                if marker_ids is not None:
                    # Access mangled private drawing functions from CameraCalibration class
                    CameraCalibration._CameraCalibration__draw_detected_markers_custom(
                        display_frame, marker_corners, marker_ids, line_thickness=2, font_scale=0.6, text_thickness=1
                    )
                # Draw subpixel corners
                if corners is not None and ids is not None:
                    CameraCalibration._CameraCalibration__draw_detected_corners_charuco_custom(
                        display_frame, corners, ids, corner_radius=6, line_thickness=1, font_scale=0.5, text_thickness=1
                    )
                    
            # Overlay info box on the display window
            info_text = f"FPS: {fps:.1f} | Detection: {'ON' if detect_mode else 'OFF'}"
            cv2.rectangle(display_frame, (10, 10), (320, 45), (0, 0, 0), -1)
            cv2.putText(display_frame, info_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('d'):
                detect_mode = not detect_mode
                print(f"ChArUco detection overlay toggled: {'ENABLED' if detect_mode else 'DISABLED'}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                # Save the raw unannotated frame
                cv2.imwrite(filepath, frame)
                print(f"Saved snapshot to: '{filepath}'")
                
                # Short flash visual feedback on save
                flash = np.ones_like(display_frame) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(80)
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exited.")

if __name__ == "__main__":
    main()
