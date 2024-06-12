import itertools
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


def plot(image_name, image):
    plt.figure(figsize=(12, 12), dpi=200)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(image_name, fontsize=20)
    plt.tight_layout()
    plt.show()


def get_threshold(image, sigma=0.23):
    v = np.median(image)
    lt = int(max(0, (1.0 - sigma) * v))
    ht = int(min(255, (1.0 + sigma) * v))
    return lt, ht


def detect_edges(image, low_threshold, high_threshold):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey_image, (3, 3), 1.5)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


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


def hough_transform(image_data, edges):
    height, width = edges.shape
    diagonal = int(np.sqrt(height ** 2 + width ** 2))

    ds = np.arange(-diagonal, diagonal + 1, 1)
    thetas = np.arange(0, np.pi, np.pi / 180)

    num_thetas = len(thetas)
    num_ds = len(ds)
    accumulator = np.zeros([num_ds, num_thetas], dtype=np.int64)
    voting_points = {}

    sins = np.sin(thetas)
    coss = np.cos(thetas)

    ys, xs = np.nonzero(edges)

    for x, y in zip(xs, ys):
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
    lines = [(line, len(voting_points[tuple(line)])) for line in lines]
    lines.sort(key=lambda ln: ln[1], reverse=True)
    lines = [item[0] for item in lines]

    if 'max_lines_number' in image_data.keys():
        lines = lines[:image_data['max_lines_number']]

    return lines, accumulator, voting_points, ds, thetas


def draw_lines(image, lines):
    line_img = image.copy()
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + len(image[0]) * (-b))
        y1 = int(y0 + len(image[1]) * a)
        x2 = int(x0 - len(image[0]) * (-b))
        y2 = int(y0 - len(image[1]) * a)
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
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
    for line1, line2 in itertools.combinations(lines, 2):
        pt = calculate_intersection(line1, line2)
        if pt is not None:
            x, y = pt
            # Check if the intersection is within the bounds of the image
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                # Store intersections in both possible orders to ensure easy lookup
                intersections[(line1, line2)] = (x, y)
                intersections[(line2, line1)] = (x, y)
    return intersections


def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_angle_between_lines(line1, line2):
    """ Calculate the angle in radians between two lines given in (rho, theta) format. """
    theta1, theta2 = line1[1], line2[1]
    angle = np.abs(theta1 - theta2)
    return angle if angle <= np.pi / 2 else np.pi - angle


def classify_triangle(pt1, pt2, pt3, line1, line2, line3):
    d1 = distance(pt1, pt2)
    d2 = distance(pt2, pt3)
    d3 = distance(pt3, pt1)
    angles = [
        get_angle_between_lines(line1, line2),
        get_angle_between_lines(line2, line3),
        get_angle_between_lines(line3, line1)
    ]
    # Triangle inequality check
    if not ((d1 + d2 > d3) and (d2 + d3 > d1) and (d3 + d1 > d2)):
        return 'not a triangle'
    # Classify as right triangle if any angle is close to 90 degrees
    if any(np.isclose(ang, np.pi / 2, atol=np.pi / 90) for ang in angles):
        return 'right'
    # Equilateral triangle check (all sides approximately equal)
    elif np.isclose(d1, d2, atol=2) and np.isclose(d2, d3, atol=2) and np.isclose(d1, d3, atol=2) and all(
            np.isclose(ang, np.pi / 3, atol=0.1) for ang in angles):
        return 'equilateral'
    # Isosceles triangle check (at least two sides approximately equal)
    elif np.isclose(d1, d2, atol=2) or np.isclose(d2, d3, atol=2) or np.isclose(d3, d1, atol=2):
        return 'isosceles'
    else:
        return 'other'


def find_and_classify_triangles(intersections, lines):
    triangles = {'isosceles': [], 'right': [], 'equilateral': []}
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
                    if pt1 != pt2 and pt2 != pt3 and pt3 != pt1:
                        triangle_type = classify_triangle(pt1, pt2, pt3, line1, line2, line3)
                        if triangle_type != 'other' and triangle_type != 'not a triangle':
                            triangles[triangle_type].append((pt1, pt2, pt3))
                            triangle_lines[triangle_type].add((line1, pt3, pt1))
                            triangle_lines[triangle_type].add((line2, pt1, pt2))
                            triangle_lines[triangle_type].add((line3, pt2, pt3))

    return triangles, triangle_lines


def line_points(start, end):
    points = []
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))

    x_inc = dx / steps
    y_inc = dy / steps

    x, y = x1, y1
    for _ in range(int(steps) + 1):
        points.append((int(round(x)), int(round(y))))
        x += x_inc
        y += y_inc

    return points


def color_edges_by_triangle(final_image, canny_edges, triangle_lines, window_position):
    color_map = {
        'equilateral': (255, 0, 0),  # Blue
        'isosceles': (0, 255, 0),  # Green
        'right': (0, 0, 255),  # Red
    }
    height, width = canny_edges.shape
    y_offset, x_offset = window_position

    for triangle_type, lines in triangle_lines.items():
        for line_info in lines:
            line_start = line_info[1]
            line_end = line_info[2]
            points = line_points(line_start, line_end)
            for x, y in points:
                nx, ny = x + x_offset, y + y_offset
                if 0 <= ny < final_image.shape[0] and 0 <= nx < final_image.shape[1]:
                    final_image[ny, nx] = color_map[triangle_type]
                    # color neighbors if they are within edge boundaries and white
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            adj_y, adj_x = ny + dy, nx + dx
                            if (0 <= adj_y < final_image.shape[0] and 0 <= adj_x < final_image.shape[1] and
                                    0 <= y + dy < height and 0 <= x + dx < width and
                                    canny_edges[y + dy, x + dx] == 255):
                                final_image[adj_y, adj_x] = color_map[triangle_type]
    return final_image


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield x, y, image[y: min(y + window_size[1], image.shape[0]), x: min(x + window_size[0], image.shape[1])]


def draw_markers(image, line, ds, thetas, color):
    rho = line[0]
    theta = line[1]
    rho_index = np.where(ds == rho)[0]
    theta_index = np.where(thetas == theta)[0]
    if rho_index.size > 0 and theta_index.size > 0:
        cv2.rectangle(image, (theta_index[0] - 5, rho_index[0] - 5), (theta_index[0] + 5, rho_index[0] + 5), color,
                      thickness=2, lineType=cv2.LINE_8)


def build_images_dict():
    images_dict = {}

    # flags2
    image = cv2.imread('group_flags/flags2.jpg')
    low_threshold, high_threshold = get_threshold(image)
    image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
                  'edge_detection_threshold': 80, 'd_threshold': 30, 'max_lines_number': 13,
                  'theta_threshold': np.pi / 30, 'window_shape': (185, 250), 'step_shape': (60, 120)}
    images_dict['flags2'] = (image, image_data)

    # overlapping-triangles-with-screwdriven-holes-wood-art11
    image = cv2.imread('group_natural/overlapping-triangles-with-screwdriven-holes-wood-art11.jpg')
    low_threshold, high_threshold = get_threshold(image)
    image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
                  'edge_detection_threshold': 50, 'd_threshold': 30, 'max_lines_number': 6,
                  'theta_threshold': np.pi / 6, 'window_shape': (180, 410), 'step_shape': (20, 25)}
    images_dict['overlapping-triangles-with-screwdriven-holes-wood-art11'] = (image, image_data)

    # t_signs1
    image = cv2.imread('group_signs/t_signs1.jpg')
    low_threshold, high_threshold = get_threshold(image)
    image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
                  'edge_detection_threshold': 110, 'd_threshold': 0.5, 'max_lines_number': 9,
                  'theta_threshold': np.pi / 30, 'window_shape': (275, 250), 'step_shape': (75, 30)}
    images_dict['t_signs1'] = (image, image_data)

    # several-triangles
    image = cv2.imread('group_sketch/several-triangles.jpg')
    low_threshold, high_threshold = get_threshold(image)
    image_data = {'canny_low_threshold': low_threshold, 'canny_high_threshold': high_threshold,
                  'edge_detection_threshold': 50, 'd_threshold': 8, 'max_lines_number': 4,
                  'theta_threshold': np.pi / 6, 'window_shape': (125, 110), 'step_shape': (25, 40)}
    images_dict['several-triangles'] = (image, image_data)

    return images_dict


def draw_accumulator(accumulator, triangle_lines, ds, thetas):
    norm_accumulator = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    norm_accumulator = cv2.cvtColor(norm_accumulator, cv2.COLOR_GRAY2BGR)

    # Convert sets to lists to access by index
    equilateral_lines = list(triangle_lines['equilateral'])
    isosceles_lines = list(triangle_lines['isosceles'])
    right_lines = list(triangle_lines['right'])

    # Draw markers for different types of triangle sides
    if equilateral_lines:
        for line_data in equilateral_lines:
            draw_markers(norm_accumulator, line_data[0], ds, thetas, (255, 0, 0))  # Blue for equilateral
    if isosceles_lines:
        for line_data in isosceles_lines:
            draw_markers(norm_accumulator, line_data[0], ds, thetas, (0, 255, 0))  # Green for isosceles
    if right_lines:
        for line_data in right_lines:
            draw_markers(norm_accumulator, line_data[0], ds, thetas, (0, 0, 255))  # Red for right

    return norm_accumulator


def find_triangles(images):
    for img_name, img in images.items():
        plot(f'{img_name} input image', img[0])

        canny_edges = detect_edges(img[0], img[1]['canny_high_threshold'], img[1]['canny_high_threshold'])
        plot(f'{img_name} edges map', canny_edges)

        final_image = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

        canny_edges_windows = sliding_window_view(canny_edges, (img[1]['window_shape'][0], img[1]['window_shape'][1]))

        for i in range(0, canny_edges_windows.shape[0], img[1]['step_shape'][0]):
            for j in range(0, canny_edges_windows.shape[1], img[1]['step_shape'][1]):
                window = canny_edges_windows[i, j]
                lines, accumulator, voting_points, ds, thetas = hough_transform(img[1], window)

                intersections = find_intersections(lines, window.shape)

                if len(intersections.keys()) >= 6:
                    triangles, triangle_lines = find_and_classify_triangles(intersections, lines)
                    if any(len(triangle_list) > 0 for triangle_list in triangles.values()):
                        lines_img = draw_lines(window.copy(), lines)
                        plot(f'{img_name} detected lines\n window_{i}_{j}', lines_img)

                        norm_accumulator = draw_accumulator(accumulator, triangle_lines, ds, thetas)
                        plot(f'{img_name}\n hough transform (with color-coded triangle sides)\n window_{i}_{j}',
                             norm_accumulator)

                        final_image = color_edges_by_triangle(final_image, window, triangle_lines, (i, j))

        plot(f'{img_name} Detected triangles (color-coded)', final_image)


images_data = build_images_dict()
find_triangles(images_data)
