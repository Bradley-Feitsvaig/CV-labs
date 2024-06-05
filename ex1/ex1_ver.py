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
    blurred = cv2.GaussianBlur(grey_image, (3, 3), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def build_images_dict(image_paths):
    images_dict = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        low_threshold, high_threshold = get_threshold(image)
        image_data = {'canny_low_threshold': low_threshold,
                      'canny_high_threshold': high_threshold,
                      'ds_steps': 10,
                      'thetas_steps': np.pi / 360,
                      'hough_threshold': 350}
        images_dict[image_path.split('.')[0]] = (image, image_data)
    return images_dict


def non_maximum_suppression(lines, rho_threshold=10, theta_threshold=10):
    suppressed_lines = []
    for rho, theta in lines:
        keep = True
        for r, t in suppressed_lines:
            if abs(rho - r) < rho_threshold and abs(theta - t) < theta_threshold:
                keep = False
                break
        if keep:
            suppressed_lines.append((rho, theta))
    return suppressed_lines


def hough_transform(image_data, edges, min_theta=-np.pi / 2, max_theta=np.pi / 2):
    # Initialize the counter matrix in polar coordinates
    height, width = edges.shape
    diagonal = int(np.sqrt(height ** 2 + width ** 2))

    ds = np.arange(-diagonal, diagonal, image_data['ds_steps'])
    thetas = np.arange(min_theta, max_theta, image_data['thetas_steps'])

    # Compute the dimension of the accumulator matrix
    num_thetas = len(thetas)
    num_ds = len(ds)
    accumulator = np.zeros([num_ds, num_thetas], dtype=np.int64)
    print('Accumulator shape (rhos x thetas):' + str(accumulator.shape))
    # Pre-compute sin and cos
    sins = np.sin(thetas)
    coss = np.cos(thetas)

    # Consider edges only
    ys, xs = np.nonzero(edges)

    for x, y in tqdm(zip(xs, ys)):
        for t in range(num_thetas):
            # compute the rhos for the given point for each theta
            current_d = int(x * coss[t] + y * sins[t])
            # for each rho, compute the closest rho among the rho_values below it
            # the index corresponding to that rho is the one we will increase
            rho_pos = np.where(current_d > ds)[0][-1]
            # rho_pos = np.argmin(np.abs(current_rho - rho_values))
            accumulator[rho_pos, t] += 1
    # Take the polar coordinates most matched
    final_rho_index, final_theta_index = np.where(accumulator > image_data['hough_threshold'])
    final_rho = ds[final_rho_index]
    final_theta = thetas[final_theta_index]

    lines = np.vstack([final_rho, final_theta]).T
    lines = non_maximum_suppression(lines)
    return lines, accumulator


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
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 80), 2)
    return line_img


image_paths = [
    'four_triangles_example.png',
    # 'group_flags/flags1.jpg',
    # 'group_natural/top-view-triangle-sandwiches-slate-with-tomatoes_23-2148640143.png',
    # 'group_signs/t_signs2.jpg',
    # 'group_sketch/several-triangles.jpg'
]

images = build_images_dict(image_paths)
for img_name, img in images.items():
    plot(img_name, img[0])
    canny_edges = detect_edges(img[0], img[1]['canny_high_threshold'], img[1]['canny_high_threshold'])
    plot(f'{img_name}_edges', canny_edges)
    lines, accumulator = hough_transform(img[1], canny_edges)

    # Normalize accumulator for visualization
    norm_accumulator = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    plot(f'{img_name}_hough', norm_accumulator)
    lines_img = draw_lines(img[0], lines)
    plot(f'{img_name} Detected Lines', lines_img)
