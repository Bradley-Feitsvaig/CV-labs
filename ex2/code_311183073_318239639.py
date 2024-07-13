import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


def load_data(image1_path, image2_path, k_matrix_path):
    """
    Load images and calibration matrix from input paths.
    :param image1_path: path to first image file.
    :param image2_path: path to second image file.
    :param k_matrix_path: path to txt file containing the calibration matrix.
    :return: Tuple with:
             - image1: The first image (np.ndarray).
             - image2: The second image (np.ndarray).
             - k: calibration matrix (np.ndarray).
    """
    # Checks if all files exist, raises exception if not
    if not os.path.isfile(image1_path):
        raise FileNotFoundError(f"{image1_path}' not found. Please check inputs.")
    if not os.path.isfile(image2_path):
        raise FileNotFoundError(f"{image2_path}' not found. Please check inputs.")
    if not os.path.isfile(k_matrix_path):
        raise FileNotFoundError(f"{k_matrix_path}' not found. Please check inputs.")

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    k = np.loadtxt(k_matrix_path, delimiter=',')
    return image1, image2, k


def find_key_points(image):
    """
    Detect key points and compute descriptors using SIFT for an input image.
    :param image: Input image (np.ndarray).
    :return: Tuple with:
             - key_points: Detected key_points (tuple of cv2.KeyPoint).
             - descriptors: Descriptors for the detected key_points (np.ndarray (#key_points X 128 features)).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    key_points, descriptors = sift.detectAndCompute(gray_image, None)
    return key_points, descriptors


def plot_key_points(image1, key_points1, image2, key_points2, images_header):
    """
    Plot the key_points of the images.
    :param image1: first image (np.ndarray).
    :param key_points1: first image key_point (tuple of cv2.KeyPoint).
    :param image2: second image (np.ndarray).
    :param key_points2: second image key_point (tuple of cv2.KeyPoint).
    :param images_header: base header for the images.
    """
    image1_with_key_points = cv2.drawKeypoints(image1, key_points1, None, color=(0, 0, 255))
    image2_with_key_points = cv2.drawKeypoints(image2, key_points2, None, color=(0, 0, 255))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1_with_key_points, cv2.COLOR_BGR2RGB))
    plt.title(f'{images_header} in Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2_with_key_points, cv2.COLOR_BGR2RGB))
    plt.title(f'{images_header} in Image 2')

    plt.show()


def find_potential_matches(descriptors1, descriptors2):
    """
    Find potential matches between two key point descriptors of 2 images,
    using BFMatcher.knnMatch() to get only 2 best matches for each descriptor.
    Applying ratio test to filtering ambiguous matches, (uses threshold of 0.8).
    :param descriptors1: Descriptors image1 key_points (np.ndarray (#key_points X 128 features))
    :param descriptors2: Descriptors image2 key_points (np.ndarray (#key_points X 128 features))
    :return: List of potential matches between descriptors1 and descriptors2.
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test
    best_matches = []
    for best_match, second_best_match in matches:
        if best_match.distance < 0.8 * second_best_match.distance:
            best_matches.append(best_match)

    return best_matches


def get_matched_key_points(matches, key_points1, key_points2):
    """
    Get the matched key points.
    :param matches: list of matches between 2 images.
    :param key_points1: first image key_point (tuple of cv2.KeyPoint).
    :param key_points2: second image key_point (tuple of cv2.KeyPoint).
    :return: Tuple with:
         - matched_key_points1: matched key_points from key_points1(tuple of cv2.KeyPoint).
         - matched_key_points2: matched key_points from key_points2(tuple of cv2.KeyPoint).
    """
    matched_key_points1 = tuple(key_points1[match.queryIdx] for match in matches)
    matched_key_points2 = tuple(key_points2[match.trainIdx] for match in matches)
    return matched_key_points1, matched_key_points2


def draw_dashed_line(image, start_point, end_point):
    """
    Draw a dashed line between start and end points.
    :param image: Image (np.ndarray).
    :param start_point: Start point of the line.
    :param end_point: End point of the line.
    """
    dist = ((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) ** 0.5
    points = []
    for i in np.arange(0, dist, 10):
        r = i / dist
        x = int((start_point[0] * (1 - r) + end_point[0] * r) + 0.5)
        y = int((start_point[1] * (1 - r) + end_point[1] * r) + 0.5)
        points.append((x, y))
    for i in range(len(points) - 1):
        if i % 2 == 0:
            cv2.line(image, points[i], points[i + 1], color=(0, 255, 255), thickness=1)


def plot_matches(image1, key_points1, image2, key_points2, matches, plot_title):
    """
    Plot the matches between two images.
    :param image1: First image (np.ndarray).
    :param key_points1: Key points of the first image (tuple of cv2.KeyPoint).
    :param image2: Second image (np.ndarray).
    :param key_points2: Key points of the second image (tuple of cv2.KeyPoint).
    :param matches: List of matches (list of cv2.DMatch).
    :param plot_title: title for the plot
    """
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    output_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype='uint8')
    output_image[:height1, :width1] = image1
    output_image[:height2, width1:] = image2

    for match in matches:
        # Matching key_points
        kp1 = key_points1[match.queryIdx].pt
        kp2 = key_points2[match.trainIdx].pt

        # Convert points to integer
        pt1 = (int(kp1[0]), int(kp1[1]))
        pt2 = (int(kp2[0]) + width1, int(kp2[1]))

        cv2.circle(output_image, pt1, 5, (0, 0, 255), 1)  # Red circles on the left key_points
        cv2.circle(output_image, pt2, 5, (0, 255, 0), 1)  # Green circles on the right key_points

        # Draw yellow dashed lines between key_points
        draw_dashed_line(output_image, pt1, pt2)

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(plot_title)
    plt.show()


def get_essential_matrix(matched_key_points1, matched_key_points2, k):
    """
    Get the Essential matrix from matched key_points and calibration matrix
    :param matched_key_points1: matched key_points from key_points1(tuple of cv2.KeyPoint).
    :param matched_key_points2:  matched key_points from key_points2(tuple of cv2.KeyPoint).
    :param k: calibration matrix (np.ndarray)
    :return: essential_matrix E (np.ndarray) and mask of inliers (np.ndarray).
    """
    matched_key_points1 = np.array([matched_key_point.pt for matched_key_point in matched_key_points1])
    matched_key_points2 = np.array([matched_key_point.pt for matched_key_point in matched_key_points2])
    E, mask = cv2.findEssentialMat(matched_key_points1, matched_key_points2, k, method=cv2.RANSAC, prob=0.9999,
                                   threshold=20.0)
    return E, mask


def get_fundamental_matrix_from_essential(E, k):
    """
    Get the fundamental matrix from the essential matrix and calibration matrix.
    :param E: essential_matrix (np.ndarray).
    :param k: calibration matrix (np.ndarray).
    :return: fundamental matrix (np.ndarray).
    """
    k_inv = np.linalg.inv(k)
    F = k_inv.T @ E @ k_inv
    return F


def filter_matches_with_essential_matrix(matches, mask):
    """
    Filter matches based on the essential matrix inliers.
    :param matches: List of matches between key points.
    :param mask: Mask of inliers from essential matrix computation.
    :return: Filtered list of matches.
    """
    inlier_matches = [match for match, inlier in zip(matches, mask.ravel()) if inlier]
    return inlier_matches


def get_epipolar_lines(points, image_index, F):
    """
    Get Epipolar lines for the input points.
    :param points: inlier matched points.
    :param image_index: index of the input image (1/2).
    :param F: fundamental matrix (np.ndarray)
    :return: epipolar_lines.
    """
    lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), image_index, F)
    lines = lines.reshape(-1, 3)
    return lines


def draw_epipolar_lines(image, epipolar_lines, points, colors):
    """
    Draw epilines on input image.
    :param image: Image on which to draw the epipolar_lines.
    :param epipolar_lines: epipolar lines to draw.
    :param points: Points from which the epipolar lines computed.
    :param colors: list of color tuples to use for drawing epipolar lines.
    """
    r, c = image.shape[:2]
    for r, pt, color in zip(epipolar_lines, points, colors):
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        image = cv2.line(image, (x0, y0), (x1, y1), color, 1)
        pt = (int(pt[0]), int(pt[1]))
        image = cv2.circle(image, tuple(pt), 5, (0, 255, 0), -1)
    return image


def visualize_epipolar_lines(image1, image2, inlier_points1, inlier_points2, F):
    """
    Visualize the epilines on both images.
    :param image1: First image (np.ndarray).
    :param image2: Second image (np.ndarray).
    :param inlier_points1: Inlier points from the first image.
    :param inlier_points2: Inlier points from the second image.
    :param F: Fundamental matrix.
    """
    inlier_points1 = np.array([kp.pt for kp in inlier_points1])
    inlier_points2 = np.array([kp.pt for kp in inlier_points2])

    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(inlier_points1))]

    lines1 = get_epipolar_lines(inlier_points2, 2, F)
    img1_epilines = draw_epipolar_lines(image1, lines1, inlier_points1, colors)

    lines2 = get_epipolar_lines(inlier_points1, 1, F)
    img2_epilines = draw_epipolar_lines(image2, lines2, inlier_points2, colors)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_epilines, cv2.COLOR_BGR2RGB))
    plt.title('Inliers and Epipolar Lines in First Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_epilines, cv2.COLOR_BGR2RGB))
    plt.title('Inliers and Epipolar Lines in Second Image')
    plt.show()


def main():
    # Read images and calibration matrix.
    image1, image2, k = load_data('data/example_1/I1.png', 'data/example_1/I2.png', 'data/example_1/K.txt')

    # 1) Find key_points in 2 images and plot them.
    image1_key_points, image1_descriptors = find_key_points(image1)
    image2_key_points, image2_descriptors = find_key_points(image2)
    plot_key_points(image1, image1_key_points, image2, image2_key_points, 'Key_points')

    # 2) Find matches between images and plot them.
    matches = find_potential_matches(image1_descriptors, image2_descriptors)
    matched_key_points1, matched_key_points2 = get_matched_key_points(matches, image1_key_points, image2_key_points)
    plot_key_points(image1, matched_key_points1, image2, matched_key_points2, 'matched key_points')
    random_matches_sample = random.sample(matches, k=70)
    plot_matches(image1, image1_key_points, image2, image2_key_points, random_matches_sample, 'matched key_points')

    # 3) Compute essential and fundamental matrix
    E, mask = get_essential_matrix(matched_key_points1, matched_key_points2, k)
    F = get_fundamental_matrix_from_essential(E, k)

    # 3 Visualize) Visualize matched key_points after outliers filter, Epipolar lines for matches inlier, Print E&F
    inlier_random_matches_sample = filter_matches_with_essential_matrix(random_matches_sample, mask)
    plot_matches(image1, image1_key_points, image2, image2_key_points, inlier_random_matches_sample,
                 'matched key_points after outliers filter')
    inlier_keypoints1, inlier_keypoints2 = get_matched_key_points(inlier_random_matches_sample, image1_key_points,
                                                                  image2_key_points)
    visualize_epipolar_lines(image1, image2, inlier_keypoints1, inlier_keypoints2, F)
    print("E:\n", E)
    print("F:\n", F)


if __name__ == "__main__":
    main()
