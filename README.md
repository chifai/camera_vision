# ChArUco Camera Calibration & Vision Pipeline

This repository contains a complete pipeline for generating ChArUco calibration boards, performing single-image or multi-image camera calibration, calculating lens distortion and perspective rectification, and verifying rectification quality using two-pass subpixel corner analysis.

---

## 1. Environment Setup

This project requires Python 3.8+ (tested on Python 3.10+). Follow these steps to set up a clean Python virtual environment and install the required dependencies.

### Windows (PowerShell)
```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS (Bash)
```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies
All required libraries are listed in `requirements.txt`:
* `numpy`: Matrix and vector computations.
* `opencv-contrib-python`: Core computer vision library including ArUco and ChArUco detector modules.
* `scipy`: Optimization and geometry conversions.
* `svgwrite`: Vector generation for calibration boards.

---

## 2. File Directory & Description

### Main Pipeline Scripts

* **[`detect_charuco.py`](file:///C:/Users/mwong/source/repos/camera_vision/detect_charuco.py)**
  The primary entry point for analyzing a single image. It performs:
  * **Corner Detection**: Identifies ArUco markers and interpolates subpixel ChArUco corners.
  * **Distortion & Pose Solving**: Solves distortion coefficients ($k_1, k_2, p_1, p_2, k_3$) and board pose (depth, Pitch/Yaw/Roll orientation).
  * **Metric Calculation**: Measures horizontal/vertical pixel-to-mm grid scales, collinearity straightness, and reprojection error.
  * **Perspective Rectification**: Generates a flat, overhead view of the board plane.
  * **Two-Pass Verification**: Runs a second detection pass automatically on the rectified image to measure the residual distortion and verify rectification quality.
  * **Outputs**: All outputs are exported to a timestamped subfolder under `./output/` (contains annotated, rectified, side-by-side comparison, and distance grid plot images, as well as a JSON summary and a CSV corner coordinates file).

* **[`acquire_and_calibrate.py`](file:///C:/Users/mwong/source/repos/camera_vision/acquire_and_calibrate.py)**
  Used for multi-image camera calibration from a folder of calibration pictures. It detects ChArUco corners across multiple perspectives to solve the intrinsic camera matrix $K$ and distortion coefficients.

* **[`generate_calib_pic.py`](file:///C:/Users/mwong/source/repos/camera_vision/generate_calib_pic.py)**
  Generates custom printable ChArUco patterns. You can specify the number of squares, physical dimensions of squares/markers, and export them as vector SVGs or high-resolution PNGs.

---

### Utility Scripts & Helpers

* **[`generate_charucoboards.ps1`](file:///C:/Users/mwong/source/repos/camera_vision/generate_charucoboards.ps1) / `generate_charucoboards.sh`**
  Batch scripts for generating different ChArUco board layouts (e.g. $7\times5$ with 3.5mm squares, $12\times9$ with 2.0mm squares).
* **[`undistort_rectify.py`](file:///C:/Users/mwong/source/repos/camera_vision/undistort_rectify.py)**
  A standalone utility for applying previously solved camera calibration parameters to undistort and rectify images.
* **[`convert_to_png.py`](file:///C:/Users/mwong/source/repos/camera_vision/convert_to_png.py)**
  Helper script using Pillow to batch convert raw camera formats (e.g., BMP, WebP, JPEG) to PNG.
* **[`arucoUtilities.py`](file:///C:/Users/mwong/source/repos/camera_vision/arucoUtilities.py)**
  Utility functions mapping ArUco/ChArUco dictionary IDs to their OpenCV counterparts.
* **[`find_edges.py`](file:///C:/Users/mwong/source/repos/camera_vision/find_edges.py) / `sobel_finder.py`**
  Helper modules used to perform Sobel and Canny edge analysis on raw calibration targets.

---

## 3. Usage Examples

### 1. Generating a Printable Board
Generate a $7\times5$ board with $3.5\text{ mm}$ square size and $2.0\text{ mm}$ marker size:
```bash
python generate_calib_pic.py --squaresX=7 --squaresY=5 --squareLength=0.0035 --markerLength=0.0020
```

### 2. Running Calibration and Rectification (Single Image)
Analyze a raw image `.\raw\leftCameraBuffer.png` using a $7\times5$ grid target, utilizing high-precision AprilTag subpixel refinement and board-layout marker recovery:
```bash
python detect_charuco.py .\raw\leftCameraBuffer.png --squares_x=7 --square_length=3.5 --corner_refinement_method=apriltag --try_refine_markers --format jpg
```

#### Key Arguments:
* `--squares_x`: Number of squares along the width of the board.
* `--square_length`: Square side length in mm.
* `--corner_refinement_method`: Choose between `none`, `subpix`, `contour`, or `apriltag`.
* `--try_refine_markers`: Search for missing markers based on known grid layout.
* `--undistort`: Corrects lens distortion along with perspective rectification (default `False`).
* `--format`: Output image format (`png` or `jpg`).
* `--output_dir`: Custom base directory for timestamped run outputs (default `./output`).

### 3. Multi-Image Intrinsics Calibration
Calibrate camera matrix intrinsics using all calibration pictures in the `.\raw\calibrate\` folder:
```bash
python acquire_and_calibrate.py .\raw\calibrate\ --squares_x=7 --square_length=3.5
```

---

## 4. Run Outputs

When running `detect_charuco.py`, a timestamped folder is generated inside `./output/` (e.g., `./output/20260715_100503_leftCameraBuffer/`):

1. **`*detected.jpg`**: Original image annotated with detected markers and corners.
2. **`*rectified.jpg`**: Perspective rectified overhead view of the ChArUco grid.
3. **`*comparison.jpg`**: Side-by-side view (left is annotated original, right is rectified).
4. **`*corners_plot.jpg`**: Connection plot mapping all detected corners with a $10\text{px}$ red circle, drawing connection lines, displaying distance measurements in mm, and listing average horizontal/vertical scaling ratios next to each row and column.
5. **`*corners.csv`**: Contains raw $(x, y)$ subpixel coordinates for each detected corner ID:
   ```csv
   corner_id,x,y
   0,569.23
   1,867.20
   ```
6. **`*results.json`**: JSON summary including collinearity deviation, reprojection errors, horizontal/vertical mm/px scales, camera matrix $K$, lens distortion coefficients, orientation Euler angles, depth, and individual row/column scale ratios.
7. **`rectified_detection/` Subfolder**: Contains the second-pass corners plot, corners CSV, and results JSON generated from running the detector on the rectified image (`*rectified.jpg`). This second-pass allows you to verify that the grid is flat and parallel ($0^\circ$ pitch/yaw/roll and collinearity errors close to $0$).
