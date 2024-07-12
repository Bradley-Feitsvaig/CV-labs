import os

import cv2
import numpy as np


def load_data(image1_path, image2_path, k_matrix_path):
    """
    Load images and calibration matrix from input paths.
    :param image1_path: path to first image file.
    :param image2_path: path to second image file.
    :param k_matrix_path: path to txt file containing the calibration matrix.
    :return: Tuple with 2 images loaded and the calibration matrix (3 np.ndarray)
    """
    # Check if all files exist, raise exception if not
    if not os.path.isfile(image1_path):
        raise FileNotFoundError(f"{image1_path}' not found. Please check inputs.")
    if not os.path.isfile(image2_path):
        raise FileNotFoundError(f"{image2_path}' not found. Please check inputs.")
    if not os.path.isfile(k_matrix_path):
        raise FileNotFoundError(f"{k_matrix_path}' not found. Please check inputs.")

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    k = np.loadtxt(k_matrix_path, delimiter=',')
    return img1, img2, k


def main():
    img1, img2, k = load_data('data/example_1/I1.png', 'data/example_1/I2.png', 'data/example_1/K.txt')


if __name__ == "__main__":
    main()
