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


def plot_matches(image1, key_points1, image2, key_points2, matches):
    """
    Plot the matches between two images.
    :param image1: First image (np.ndarray).
    :param key_points1: Key points of the first image (tuple of cv2.KeyPoint).
    :param image2: Second image (np.ndarray).
    :param key_points2: Key points of the second image (tuple of cv2.KeyPoint).
    :param matches: List of matches (list of cv2.DMatch).
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
    plt.title('matched key_points')
    plt.show()


def main():
    # Read images and calibration matrix.
    image1, image2, k = load_data('data/example_1/I1.png', 'data/example_1/I2.png', 'data/example_1/K.txt')

    # Find key_points in 2 images and plot them.
    image1_key_points, image1_descriptors = find_key_points(image1)
    image2_key_points, image2_descriptors = find_key_points(image2)
    plot_key_points(image1, image1_key_points, image2, image2_key_points, 'Key_points')

    # Find matches between images and plot them.
    matches = find_potential_matches(image1_descriptors, image2_descriptors)
    matched_key_points1, matched_key_points2 = get_matched_key_points(matches, image1_key_points, image2_key_points)
    plot_key_points(image1, matched_key_points1, image2, matched_key_points2, 'matched key_points')
    random_matches_sample = random.sample(matches, k=70)
    plot_matches(image1, image1_key_points, image2, image2_key_points, random_matches_sample)


if __name__ == "__main__":
    main()
