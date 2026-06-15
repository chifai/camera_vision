import cv2
import numpy as np
import math
import sys
import os

def polar_to_points(rho, theta):
    """Converts polar coordinates (rho, theta) to two points for drawing."""
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 5000 * (-b)), int(y0 + 5000 * (a)))
    pt2 = (int(x0 - 5000 * (-b)), int(y0 - 5000 * (a)))
    return pt1, pt2

def intersect_polar(l1, l2):
    """Finds the intersection point of two lines in polar coordinates."""
    rho1, theta1 = l1
    rho2, theta2 = l2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ], dtype=np.float64)
    B = np.array([rho1, rho2], dtype=np.float64)
    try:
        res = np.linalg.solve(A, B)
        return int(round(res[0])), int(round(res[1]))
    except np.linalg.LinAlgError:
        return None

def average_lines(lines, target_theta):
    """Averages all lines, aligning their theta and rho to a target direction."""
    if not lines: return None
    rhos = []
    thetas = []
    for rho, theta in lines:
        diff = theta - target_theta
        # Bring diff to [-pi/2, pi/2] bounds by flipping the line vector if needed
        while diff > math.pi/2:
            diff -= math.pi
            theta -= math.pi
            rho = -rho
        while diff < -math.pi/2:
            diff += math.pi
            theta += math.pi
            rho = -rho
        rhos.append(rho)
        thetas.append(theta)
    return np.mean(rhos), np.mean(thetas)


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'TSMC_cam0.png'
        
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return
        
    h, w = img.shape[:2]
    print(f"pixel: {h}, {w}")
    
    # === STEP 1. Blur the image to reduce noise ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Optional: CLAHE helps pop edges in low contrast areas
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    blurred = cv2.GaussianBlur(cl, (7, 7), 0)
    
    # === STEP 2. Do horizontal & vertical Sobel algo and calculate the magnitude ===
    # Use CV_64F to prevent overflow when calculating derivatives
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    
    # === STEP 3. When the magnitude is greater than a threshold, mark it as white ===
    # Using 97th percentile ensures we only keep the absolute strongest gradients
    threshold_val = np.percentile(mag, 99)
    mask = (mag > threshold_val).astype(np.uint8) * 255
    
    # === STEP 4. Output the processed image as debug image ===
    debug_dir = "processed/debug_sobel"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"{base_name}_sobel_debug.png")
    cv2.imwrite(debug_path, mask)
    
    # === STEP 5. Find two perpendicular edges and overlay ===
    # Use finer resolution (np.pi/1800) for sub-degree (0.1 deg) precision
    lines = cv2.HoughLines(mask, 1, np.pi/180, 500)
    if lines is None:
        print(f"No strong lines found in {image_path}")
        return
    
    print(f"len: {lines.size}")
    print(lines)
    # Group by theta to find the dominant angle
    angle_clusters = {}
    for line in lines:
        theta = line[0][1]
        deg_float = math.degrees(theta)
        
        # Filter out 45-deg artifacts caused by thick pixelated masks. 
        # True tilt is very small, so we only consider lines near 0/180 or 90.
        if not ((deg_float < 15 or deg_float > 165) or (75 < deg_float < 105)):
            continue
            
        mdeg = int(round(deg_float * 1000)) % 180000
        
        # Group similar mdeg values to avoid splitting a single peak
        added = False
        for key_mdeg in angle_clusters.keys():
            if abs(mdeg - key_mdeg) < 5000 or abs(mdeg - key_mdeg) > 175000:
                angle_clusters[key_mdeg].append(theta)
                added = True
                break
        if not added:
            angle_clusters[mdeg] = [theta]

    if not angle_clusters:
        print(f"No valid axis-aligned lines found in {image_path}")
        return
    

    # Find the most prominent angle (bin with most lines)
    best_bin = max(angle_clusters, key=lambda k: len(angle_clusters[k]))
    best_base_theta = np.mean(angle_clusters[best_bin])
    print(best_base_theta)
    
    # Enforce perpendicularity to gather the lines properly
    orthogonal_theta = (best_base_theta + math.pi/2) % math.pi

    # Gather lines for these two fixed perpendicular angles
    group1_lines, group2_lines = [], []
    for line in lines:
        rho, theta = line[0]
        if abs(theta - best_base_theta) < math.radians(10) or abs(theta - best_base_theta + math.pi) < math.radians(10):
            group1_lines.append((rho, theta))
        elif abs(theta - orthogonal_theta) < math.radians(10) or abs(theta - orthogonal_theta + math.pi) < math.radians(10):
            group2_lines.append((rho, theta))

    if not group1_lines or not group2_lines:
        print(f"Failed to find two perpendicular orientations in {image_path}")
        return

    # Determine which is H and V
    if abs(best_base_theta - math.pi/2) < abs(orthogonal_theta - math.pi/2):
        theta_h, theta_v = best_base_theta, orthogonal_theta
        lines_h, lines_v = group1_lines, group2_lines
    else:
        theta_h, theta_v = orthogonal_theta, best_base_theta
        lines_h, lines_v = group2_lines, group1_lines

    # Average all horizontal and vertical lines
    line_h = average_lines(lines_h, theta_h)
    line_v = average_lines(lines_v, theta_v)
    
    if line_h is None or line_v is None:
        print(f"Could not calculate average edges in {image_path}")
        return

    # Intersection
    intersect = intersect_polar(line_h, line_v)
    overlay = img.copy()

    # Draw lines
    p1_1, p1_2 = polar_to_points(*line_h)
    p2_1, p2_2 = polar_to_points(*line_v)
    cv2.line(overlay, p1_1, p1_2, (0, 255, 0), 6) # H - Green
    cv2.line(overlay, p2_1, p2_2, (255, 0, 0), 6) # V - Blue

    # Draw intersection and text
    if intersect:
        cv2.circle(overlay, intersect, 10, (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"Int: {intersect}", (intersect[0]+70, intersect[1]-70), font, 2.5, (255, 255, 255), 5)
        
        h_tilt = math.degrees(line_h[1]) - 90
        v_tilt = math.degrees(line_v[1])
        if v_tilt >= 90: v_tilt -= 180
        cv2.putText(overlay, f"H-Tilt: {h_tilt:.2f} deg", (50, 90), font, 2.2, (0, 255, 0), 5)
        cv2.putText(overlay, f"V-Tilt: {v_tilt:.2f} deg", (50, 180), font, 2.2, (255, 0, 0), 5)

    edges_dir = "processed/edges_sobel"
    os.makedirs(edges_dir, exist_ok=True)
    out_path = os.path.join(edges_dir, f"{base_name}_sobel_edges.png")
    cv2.imwrite(out_path, overlay)
    print(f"Processed {image_path}. Intersection: {intersect}")

if __name__ == "__main__":
    main()