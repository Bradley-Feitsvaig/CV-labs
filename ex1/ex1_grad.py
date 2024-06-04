import numpy as np
import cv2
from matplotlib import pyplot as plt
from itertools import combinations
from tqdm import tqdm

# Load the input images
image_paths = ['four_triangles_example.png']
images = [cv2.imread(image_path) for image_path in image_paths]


def detect_edges(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    return edges


def compute_gradient(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction


def hough_transform(edges, gradient_direction):
    height, width = edges.shape
    diag_len = int(np.sqrt(height ** 2 + width ** 2))
    ds = np.arange(-diag_len, diag_len, 1)
    thetas = np.deg2rad(np.arange(-90, 90, 0.1))

    accumulator = np.zeros((len(ds), len(thetas)), dtype=np.int64)
    y_idxs, x_idxs = np.nonzero(edges)

    print("Performing Hough Transform...")
    for i in tqdm(range(len(x_idxs))):
        x = x_idxs[i]
        y = y_idxs[i]
        theta = gradient_direction[y, x]
        theta_idx = np.argmin(np.abs(thetas - theta))
        d = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
        accumulator[d, theta_idx] += 1

    lines = []
    print("Extracting lines from accumulator...")
    for d_idx in tqdm(range(accumulator.shape[0])):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[d_idx, theta_idx] > 150:
                d = ds[d_idx]
                theta = thetas[theta_idx]
                lines.append((d, theta))

    print(f"Detected {len(lines)} lines")
    return lines, accumulator, thetas, ds


def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    cos_theta1, sin_theta1 = np.cos(theta1), np.sin(theta1)
    cos_theta2, sin_theta2 = np.cos(theta2), np.sin(theta2)

    det = cos_theta1 * sin_theta2 - cos_theta2 * sin_theta1
    if np.abs(det) < 1e-10:
        return None

    x = (rho2 * sin_theta1 - rho1 * sin_theta2) / det
    y = (rho1 * cos_theta2 - rho2 * cos_theta1) / det
    return (x, y)


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def classify_triangle(sides):
    a, b, c = sorted(sides)
    if np.isclose(a, b) and np.isclose(b, c):
        return 'equilateral'
    elif np.isclose(a, b) or np.isclose(b, c) or np.isclose(a, c):
        return 'isosceles'
    elif np.isclose(a ** 2 + b ** 2, c ** 2):
        return 'right'
    else:
        return 'scalene'


def detect_triangles(lines, width, height):
    equilateral_lines = set()
    isosceles_lines = set()
    right_triangle_lines = set()

    intersections = {}
    print("Detecting intersections...")
    for line1, line2 in tqdm(combinations(lines, 2)):
        if abs(line1[1] - line2[1]) > np.deg2rad(10):
            point = line_intersection(line1, line2)
            if point and 0 <= point[0] <= width and 0 <= point[1] <= height:
                intersections[(line1, line2)] = point

    intersection_keys = list(intersections.keys())
    print(f"Found {len(intersection_keys)} intersections")
    print("Classifying triangles...")
    for i, (l1, l2) in tqdm(enumerate(intersection_keys)):
        for j in range(i + 1, len(intersection_keys)):
            l3, l4 = intersection_keys[j]
            if l1 == l3 or l1 == l4 or l2 == l3 or l2 == l4:
                for k in range(j + 1, len(intersection_keys)):
                    l5, l6 = intersection_keys[k]
                    if l1 == l5 or l1 == l6 or l2 == l5 or l2 == l6 or l3 == l5 or l3 == l6 or l4 == l5 or l4 == l6:
                        p1 = intersections[(l1, l2)]
                        p2 = intersections[(l3, l4)]
                        p3 = intersections[(l5, l6)]

                        if not (np.allclose(p1, p2) or np.allclose(p2, p3) or np.allclose(p3, p1)):
                            sides = [
                                calculate_distance(p1, p2),
                                calculate_distance(p2, p3),
                                calculate_distance(p3, p1)
                            ]
                            a, b, c = sorted(sides)
                            if a + b > c:
                                classification = classify_triangle(sides)
                                if classification == 'equilateral':
                                    equilateral_lines.update([l1, l2, l3, l4, l5, l6])
                                elif classification == 'isosceles':
                                    isosceles_lines.update([l1, l2, l3, l4, l5, l6])
                                elif classification == 'right':
                                    right_triangle_lines.update([l1, l2, l3, l4, l5, l6])

    print(
        f"Equilateral: {len(equilateral_lines)}, Isosceles: {len(isosceles_lines)}, Right: {len(right_triangle_lines)}")
    return list(equilateral_lines), list(isosceles_lines), list(right_triangle_lines)


def plot_results(image, edges, lines, equilateral_lines, isosceles_lines, right_triangle_lines, accumulator, thetas,
                 ds):
    plt.figure(figsize=(12, 12), dpi=100)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Input Image')
    plt.show()

    plt.figure(figsize=(12, 12), dpi=100)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.title('Edge Map')
    plt.show()

    line_img = image.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    plt.figure(figsize=(12, 12), dpi=100)
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected Lines')
    plt.show()

    norm_accumulator = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_accumulator = cv2.cvtColor(norm_accumulator, cv2.COLOR_GRAY2BGR)

    for line in equilateral_lines:
        rho, theta = line
        theta_idx = np.where(thetas == theta)[0][0]
        d_idx = np.where(ds == rho)[0][0]
        cv2.rectangle(color_accumulator,
                      (theta_idx - 2, d_idx - 2),
                      (theta_idx + 2, d_idx + 2),
                      (255, 0, 0), thickness=2)

    for line in isosceles_lines:
        rho, theta = line
        theta_idx = np.where(thetas == theta)[0][0]
        d_idx = np.where(ds == rho)[0][0]
        cv2.rectangle(color_accumulator,
                      (theta_idx - 2, d_idx - 2),
                      (theta_idx + 2, d_idx + 2),
                      (0, 255, 0), thickness=2)

    for line in right_triangle_lines:
        rho, theta = line
        theta_idx = np.where(thetas == theta)[0][0]
        d_idx = np.where(ds == rho)[0][0]
        cv2.rectangle(color_accumulator,
                      (theta_idx - 2, d_idx - 2),
                      (theta_idx + 2, d_idx + 2),
                      (0, 0, 255), thickness=2)

    plt.figure(figsize=(12, 12), dpi=100)
    plt.imshow(color_accumulator)
    plt.axis('off')
    plt.title('Hough Transform (with color-coded triangle sides)')
    plt.show()


for img in images:
    edges = detect_edges(img)
    gradient_magnitude, gradient_direction = compute_gradient(img)
    lines, accumulator, thetas, ds = hough_transform(edges, gradient_direction)
    height, width = edges.shape
    equilateral_lines, isosceles_lines, right_triangle_lines = detect_triangles(lines, width, height)
    plot_results(img, edges, lines, equilateral_lines, isosceles_lines, right_triangle_lines, accumulator, thetas, ds)

