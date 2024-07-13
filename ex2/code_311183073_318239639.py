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


def plot_key_points(image1, key_points1, image2, key_points2):
    """
    Plot the key_points of the images.
    :param image1: first image (np.ndarray)
    :param key_points1: first image key_point (tuple of cv2.KeyPoint)
    :param image2: second image (np.ndarray)
    :param key_points2: second image key_point (tuple of cv2.KeyPoint)
    """
    image1_with_key_points = cv2.drawKeypoints(image1, key_points1, None, color=(0, 0, 255))
    image2_with_key_points = cv2.drawKeypoints(image2, key_points2, None, color=(0, 0, 255))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1_with_key_points, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2_with_key_points, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Image 2')

    plt.show()


def find_potential_matches(descriptors1, descriptors2):
    """
    Find potential matches between two key point descriptors of 2 images,
    using BFMatcher.knnMatch() to get only 2 best matches for each descriptor.
    Applying ratio test to filtering ambiguous matches, (uses threshold of 0.8).
    :param descriptors1: Descriptors image1 key_points (np.ndarray (#key_points X 128 features))
    :param descriptors2: Descriptors image2 key_points (np.ndarray (#key_points X 128 features))
    :return: list of potential matches between descriptors1 and descriptors2.
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append([m])

    return good_matches


def get_matched_key_points(matches, key_points1, key_points2):
    """
    Get the matched key points.
    :param matches: List of matches between 2 images.
    :param key_points1: first image key_point (tuple of cv2.KeyPoint)
    :param key_points2: second image key_point (tuple of cv2.KeyPoint)
    :return: Tuple with:
         - matched_key_points1: matched key_points from key_points1(tuple of cv2.KeyPoint).
         - matched_key_points2: matched key_points from key_points2(tuple of cv2.KeyPoint).
    """
    matched_key_points1 = tuple(key_points1[m[0].queryIdx] for m in matches)
    matched_key_points2 = tuple(key_points2[m[0].trainIdx] for m in matches)
    return matched_key_points1, matched_key_points2


def plot_matches(image1, key_points1, image2, key_points2, matches):
    """
    Plot the matches between two images.
    :param image1: First image (np.ndarray).
    :param key_points1: Key points of the first image (list of cv2.KeyPoint).
    :param image2: Second image (np.ndarray).
    :param key_points2: Key points of the second image (list of cv2.KeyPoint).
    :param matches: List of matches (list of lists containing cv2.DMatch objects).
    """
    image_matches = cv2.drawMatchesKnn(image1, key_points1, image2, key_points2, matches, None,
                                       matchColor=(0, 255, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matches')
    plt.show()


def main():
    # Read images and calibration matrix.
    image1, image2, k = load_data('data/example_1/I1.png', 'data/example_1/I2.png', 'data/example_1/K.txt')
    # Find key_points in 2 images and plot them.
    image1_key_points, image1_descriptors = find_key_points(image1)
    image2_key_points, image2_descriptors = find_key_points(image2)
    plot_key_points(image1, image1_key_points, image2, image2_key_points)
    # Find matches between images and plot them.
    matches = find_potential_matches(image1_descriptors, image2_descriptors)
    matched_key_points1, matched_key_points2 = get_matched_key_points(matches, image1_key_points, image2_key_points)
    plot_key_points(image1, matched_key_points1, image2, matched_key_points2)
    random_matches_sample = random.sample(matches, k=70)
    plot_matches(image1, image1_key_points, image2, image2_key_points, random_matches_sample)


if __name__ == "__main__":
    main()
