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

def get_exterior_rho(lines, center_val, label=""):
    """Finds all significant parallel clusters and returns the rho of the outermost one."""
    if not lines: return None
    lines.sort(key=lambda x: x[0])
    
    # Cluster by rho
    clusters = []
    curr = [lines[0]]
    for i in range(1, len(lines)):
        if abs(lines[i][0] - lines[i-1][0]) < 40:
            curr.append(lines[i])
        else:
            clusters.append(curr)
            curr = [lines[i]]
    clusters.append(curr)
    
    # Summarize clusters
    summaries = []
    for c in clusters:
        summaries.append({
            'avg_rho': np.mean([l[0] for l in c]),
            'count': len(c)
        })
    
    # Sort by prominence
    summaries.sort(key=lambda x: x['count'], reverse=True)
    
    # Define 'significant' as having at least 30% of the max count
    max_count = summaries[0]['count']
    significant = [s for s in summaries if s['count'] >= max_count * 0.3]
    
    # Among significant clusters, pick the one furthest from center_val
    significant.sort(key=lambda x: abs(x['avg_rho'] - center_val), reverse=True)
    
    print(f"  {label} clusters found: {len(summaries)} (Significant: {len(significant)})")
    for i, s in enumerate(significant[:3]):
        print(f"    - Cluster {i}: rho={s['avg_rho']:.2f}, count={s['count']}, dist={abs(s['avg_rho']-center_val):.2f}")

    return significant[0]['avg_rho']

def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'TSMC_cam0.png' if os.path.exists('TSMC_cam0.png') else 'TSMC_cam0.jpg'
            
    base_name = os.path.splitext(image_path)[0]
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w = img.shape[:2]
    
    # 1. Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Stronger CLAHE for better contrast in shadows
    cl = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8)).apply(gray)
    
    # Reduced blur to preserve dual edges
    blurred = cv2.GaussianBlur(cl, (5, 5), 0)
    
    # More sensitive Canny thresholds
    edges = cv2.Canny(blurred, 25, 100)
    
    # Smaller morphological kernel to prevent merging parallel lines
    kernel = np.ones((3,3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{base_name}_debug.png", edges_closed)

    # 2. Line detection (Using Probabilistic Hough for length constraint)
    # minLineLength: Only lines longer than this are returned.
    # maxLineGap: Max distance between segments to treat them as a single line.
    segments = cv2.HoughLinesP(edges_closed, 1, np.pi/180, threshold=80, minLineLength=200, maxLineGap=100)

    if segments is None:
        return print("No lines meeting length constraint (200px) detected")

    # Convert segments [x1, y1, x2, y2] to polar [rho, theta] for our stable math
    lines = []
    for seg in segments:
        x1, y1, x2, y2 = seg[0]
        # Calculate theta using atan2 for precise direction
        angle = math.atan2(y2 - y1, x2 - x1)
        theta = (angle + math.pi/2) % math.pi

        # Distance rho: r = x*cos(theta) + y*sin(theta)
        rho = x1 * math.cos(theta) + y1 * math.sin(theta)
        lines.append([(rho, theta)])


    # Group by theta to find the most dominant angle
    angle_bins = {} # deg -> list of original thetas
    for line in lines:
        theta = line[0][1]
        deg = int(round(math.degrees(theta))) % 180
        if deg not in angle_bins:
            angle_bins[deg] = []
        angle_bins[deg].append(theta)

    # Find the most prominent angle (bin with most lines)
    best_bin = max(angle_bins, key=lambda k: len(angle_bins[k]))
    
    # PRECISION: Average the actual thetas in this bin
    best_base_theta = np.mean(angle_bins[best_bin])
    
    # PERPENDICULAR CONSTRAINT
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
        return print("Failed to find perpendicular edges.")

    # Determine which is H and V
    if abs(best_base_theta - math.pi/2) < abs(orthogonal_theta - math.pi/2):
        theta_h, theta_v = best_base_theta, orthogonal_theta
        lines_h, lines_v = group1_lines, group2_lines
    else:
        theta_h, theta_v = orthogonal_theta, best_base_theta
        lines_h, lines_v = group2_lines, group1_lines

    # TWO-LINE APPROACH: Find exterior rho from all significant parallel edges
    rho_h = get_exterior_rho(lines_h, h/2, "Horizontal")
    rho_v = get_exterior_rho(lines_v, w/2, "Vertical")

    line_h = (rho_h, theta_h)
    line_v = (rho_v, theta_v)

    intersect = intersect_polar(line_h, line_v)
    overlay = img.copy()

    # Draw
    p1_1, p1_2 = polar_to_points(*line_h)
    p2_1, p2_2 = polar_to_points(*line_v)
    cv2.line(overlay, p1_1, p1_2, (0, 255, 0), 6) # H
    cv2.line(overlay, p2_1, p2_2, (255, 0, 0), 6) # V

    if intersect:
        cv2.circle(overlay, intersect, 10, (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"Int: {intersect}", (intersect[0]+70, intersect[1]-70), font, 2.5, (255, 255, 255), 5)
        cv2.putText(overlay, f"H-Tilt: {math.degrees(theta_h)-90:.2f} deg", (50, 90), font, 2.2, (0, 255, 0), 5)
        cv2.putText(overlay, f"V-Tilt: {math.degrees(theta_v)- (0 if math.degrees(theta_v)<90 else 180):.2f} deg", (50, 180), font, 2.2, (255, 0, 0), 5)

    cv2.imwrite(f"{base_name}_edges.png", overlay)
    print(f"Processed {image_path}. Intersection: {intersect}")

if __name__ == "__main__":
    main()
