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

def get_exterior_rho(lines, center_val):
    """Finds the two most prominent parallel clusters and returns the rho of the outermost one."""
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
    
    # Summarize clusters by prominence (count)
    summaries = []
    for c in clusters:
        summaries.append({
            'avg_rho': np.mean([l[0] for l in c]),
            'count': len(c)
        })
    
    # Sort by prominence (number of lines in Hough space)
    summaries.sort(key=lambda x: x['count'], reverse=True)
    
    if len(summaries) < 2:
        return summaries[0]['avg_rho']
    
    # We have at least two prominent parallel edges (likely interior and exterior)
    line1 = summaries[0]['avg_rho']
    line2 = summaries[1]['avg_rho']
    
    # Pick the one furthest from the image center (Exterior)
    if abs(line1 - center_val) > abs(line2 - center_val):
        return line1
    else:
        return line2

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
    cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(cl, (7, 7), 0), 40, 130)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # 2. Line detection
    lines = cv2.HoughLines(edges_closed, 1, np.pi/180, 180)
    if lines is None: return print("No lines")

    # Group by theta to find the most dominant angle
    angle_bins = {}
    for line in lines:
        theta = line[0][1]
        deg = int(round(math.degrees(theta))) % 180
        angle_bins[deg] = angle_bins.get(deg, 0) + 1

    best_base_deg = max(angle_bins, key=angle_bins.get)
    best_base_theta = math.radians(best_base_deg)
    
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

    # TWO-LINE APPROACH: Find exterior rho from the two most prominent parallel edges
    rho_h = get_exterior_rho(lines_h, h/2)
    rho_v = get_exterior_rho(lines_v, w/2)

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
