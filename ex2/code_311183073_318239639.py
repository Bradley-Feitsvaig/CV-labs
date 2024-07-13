import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


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
             - descriptors: Descriptors for the detected key_points (np.ndarray).
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


def main():
    image1, image2, k = load_data('data/example_1/I1.png', 'data/example_1/I2.png', 'data/example_1/K.txt')
    image1_key_points, image1_descriptors = find_key_points(image1)
    image2_key_points, image2_descriptors = find_key_points(image2)
    plot_key_points(image1, image1_key_points, image2, image2_key_points)


if __name__ == "__main__":
    main()
