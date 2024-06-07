import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot(image_name, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(image_name)
    plt.show()


def get_threshold(image, sigma=0.23):
    v = np.median(image)
    lt = int(max(0, (1.0 - sigma) * v))
    ht = int(min(255, (1.0 + sigma) * v))
    return lt, ht


def detect_edges(image, low_threshold, high_threshold):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey_image, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def build_images_dict():
    images_dict = {}
    # four_triangles_example
    image = cv2.imread('four_triangles_example.png')
    low_threshold, high_threshold = get_threshold(image)
    image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
                  'hough_min_theta': -np.pi / 2, 'hough_max_theta': np.pi / 2, 'ds_steps': 1,
                  'thetas_steps': np.pi / 180, 'edge_detection_threshold': 150, 'd_threshold': 10,
                  'theta_threshold': 0.1}
    images_dict['four_triangles_example'] = (image, image_data)

    # # flags1
    # image = cv2.imread('group_flags/flags1.jpg')
    # low_threshold, high_threshold = get_threshold(image)
    # image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
    #               'hough_min_theta': -np.pi / 2, 'hough_max_theta': np.pi / 2, 'ds_steps': 7,
    #               'thetas_steps': np.pi / 400, 'edge_detection_threshold': 320, 'd_threshold': 30, 'theta_threshold': 2}
    # images_dict['flags1'] = (image, image_data)
    #
    # # top-view-triangle-sandwiches
    # image = cv2.imread('group_natural/top-view-triangle-sandwiches-slate-with-tomatoes_23-2148640143.png')
    # low_threshold, high_threshold = get_threshold(image)
    # image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
    #               'hough_min_theta': -np.pi / 2, 'hough_max_theta': np.pi / 2, 'ds_steps': 7,
    #               'thetas_steps': np.pi / 400, 'edge_detection_threshold': 320, 'd_threshold': 30, 'theta_threshold': 2}
    # images_dict['top-view-triangle-sandwiches-slate-with-tomatoes'] = (image, image_data)
    #
    # # t_signs2
    # image = cv2.imread('group_signs/t_signs2.jpg')
    # low_threshold, high_threshold = get_threshold(image)
    # image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
    #               'hough_min_theta': -np.pi / 2, 'hough_max_theta': np.pi / 2, 'ds_steps': 7,
    #               'thetas_steps': np.pi / 400, 'edge_detection_threshold': 320, 'd_threshold': 30, 'theta_threshold': 2}
    # images_dict['t_signs2'] = (image, image_data)
    #
    # # several-triangles
    # image = cv2.imread('group_sketch/several-triangles.jpg')
    # low_threshold, high_threshold = get_threshold(image)
    # image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
    #               'hough_min_theta': -np.pi / 2, 'hough_max_theta': np.pi / 2, 'ds_steps': 7,
    #               'thetas_steps': np.pi / 400, 'edge_detection_threshold': 320, 'd_threshold': 30, 'theta_threshold': 2}
    # images_dict['several-triangles'] = (image, image_data)

    return images_dict


def remove_close_lines(lines, d_threshold, theta_threshold):
    suppressed_lines = []
    for rho, theta in lines:
        keep = True
        for r, t in suppressed_lines:
            if abs(rho - r) < d_threshold and abs(theta - t) < theta_threshold:
                keep = False
                break
        if keep:
            suppressed_lines.append((rho, theta))
    return suppressed_lines


def hough_transform(image_data, edges, min_theta, max_theta):
    height, width = edges.shape
    diagonal = int(np.sqrt(height ** 2 + width ** 2))

    ds = np.arange(-diagonal, diagonal, image_data['ds_steps'])
    thetas = np.arange(min_theta, max_theta, image_data['thetas_steps'])

    num_thetas = len(thetas)
    num_ds = len(ds)
    accumulator = np.zeros([num_ds, num_thetas], dtype=np.int64)
    voting_points = {}

    sins = np.sin(thetas)
    coss = np.cos(thetas)

    ys, xs = np.nonzero(edges)

    for x, y in tqdm(zip(xs, ys)):
        for t in range(num_thetas):
            current_d = int(x * coss[t] + y * sins[t])
            rho_pos = np.where(current_d > ds)[0][-1]
            accumulator[rho_pos, t] += 1
            line = (ds[rho_pos], thetas[t])
            if line not in voting_points:
                voting_points[line] = []
            voting_points[line].append((x, y))

    final_rho_index, final_theta_index = np.where(accumulator > image_data['edge_detection_threshold'])
    final_rho = ds[final_rho_index]
    final_theta = thetas[final_theta_index]

    lines = np.vstack([final_rho, final_theta]).T
    lines = remove_close_lines(lines, image_data['d_threshold'], image_data['theta_threshold'])
    return lines, accumulator, voting_points


def draw_lines(image, lines):
    line_img = image.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + len(image[0]) * (-b))
        y1 = int(y0 + len(image[1]) * (a))
        x2 = int(x0 - len(image[0]) * (-b))
        y2 = int(y0 - len(image[1]) * (a))
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return line_img


def calculate_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    if np.linalg.det(A) == 0:
        return None
    solution = np.linalg.solve(A, b)
    x, y = solution
    return int(x), int(y)


def find_intersections(lines, image_shape):
    """ Calculate intersections of all line pairs and store them in a dictionary. """
    intersections = {}
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            pt = calculate_intersection(lines[i], lines[j])
            if pt is not None:
                x, y = pt
                # Check if the intersection is within the bounds of the image
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    # Store intersections in both possible orders to ensure easy lookup
                    intersections[(lines[i], lines[j])] = (x, y)
                    intersections[(lines[j], lines[i])] = (x, y)
    return intersections


def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def angle(line1, line2):
    """ Calculate the angle in radians between two lines given in (rho, theta) format. """
    theta1, theta2 = line1[1], line2[1]
    angle = np.abs(theta1 - theta2)
    return angle if angle <= np.pi / 2 else np.pi - angle


def classify_triangle(pt1, pt2, pt3, line1, line2, line3):
    d1 = distance(pt1, pt2)
    d2 = distance(pt2, pt3)
    d3 = distance(pt3, pt1)
    angles = [
        angle(line1, line2),
        angle(line2, line3),
        angle(line3, line1)
    ]

    # Classify as right triangle if any angle is close to 90 degrees
    if any(np.isclose(ang, np.pi / 2, atol=0.1) for ang in angles):  # 0.1 rad tolerance for right angle
        return 'right'
    # Equilateral triangle check (all sides approximately equal)
    elif np.isclose(d1, d2, atol=10) and np.isclose(d2, d3, atol=10):
        return 'equilateral'
    # Isosceles triangle check (at least two sides approximately equal)
    elif np.isclose(d1, d2, atol=10) or np.isclose(d2, d3, atol=10) or np.isclose(d3, d1, atol=10):
        return 'isosceles'
    else:
        return 'other'


def find_and_classify_triangles(intersections, lines):
    triangles = {'isosceles': [], 'right': [],'equilateral': []}
    triangle_lines = {'equilateral': set(), 'isosceles': set(), 'right': set()}
    n = len(lines)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                line1, line2, line3 = lines[i], lines[j], lines[k]
                if (line1, line2) in intersections and (line2, line3) in intersections and (
                        line3, line1) in intersections:
                    pt1 = intersections[(line1, line2)]
                    pt2 = intersections[(line2, line3)]
                    pt3 = intersections[(line3, line1)]
                    triangle_type = classify_triangle(pt1, pt2, pt3, line1, line2, line3)
                    if triangle_type != 'other':
                        triangles[triangle_type].append((pt1, pt2, pt3))
                        triangle_lines[triangle_type].add(line1)
                        triangle_lines[triangle_type].add(line2)
                        triangle_lines[triangle_type].add(line3)

    return triangles, triangle_lines


def color_edges_by_triangle(canny_edges, triangle_lines,voting_points):
    color_map = {
        'equilateral': (0, 0, 255),   # Blue
        'isosceles': (0, 255, 0),     # Green
        'right': (255, 0, 0),         # Red
    }

    color_edges = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    height, width = canny_edges.shape

    for triangle_type, lines in triangle_lines.items():
        for line in lines:
            points = voting_points[line]
            for x, y in points:
                color_edges[y, x] = color_map[triangle_type]
                # Color neighbors if they are white
                for dy in range(-4, 5):  # Check a 9X9 neighborhood
                    for dx in range(-4, 5):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and canny_edges[ny, nx] == 255:
                            color_edges[ny, nx] = color_map[triangle_type]

    return color_edges


images = build_images_dict()
for img_name, img in images.items():
    plot(img_name, img[0])
    canny_edges = detect_edges(img[0], img[1]['canny_high_threshold'], img[1]['canny_high_threshold'])
    plot(f'{img_name}_edges', canny_edges)
    lines, accumulator, voting_points = hough_transform(img[1], canny_edges, img[1]['hough_min_theta'], img[1]['hough_max_theta'])

    # Normalize accumulator for visualization
    norm_accumulator = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    plot(f'{img_name}_hough', norm_accumulator)
    lines_img = draw_lines(canny_edges.copy(), lines)
    plot(f'{img_name} Detected Lines', lines_img)

    intersections = find_intersections(lines, img[0].shape)

    # To visualize the intersections:
    for x, y in intersections.values():
        cv2.circle(img[0], (x, y), radius=5, color=(255, 0, 0), thickness=-1)  # Draw red dots at intersections
    plot('Intersections', img[0])
    triangles, triangle_lines = find_and_classify_triangles(intersections, lines)
    image_with_triangles = color_edges_by_triangle(canny_edges.copy(), triangle_lines,voting_points)
    plot(f'{img_name} with Triangles', image_with_triangles)

    # Print counts of each triangle type
    for triangle_type, triangle_list in triangles.items():
        print(f"{triangle_type.capitalize()} triangles found: {len(triangle_list)}")
    print(triangle_lines)
