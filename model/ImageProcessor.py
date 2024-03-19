#!/opt/homebrew/bin/python3
"""
Name: ImageProcessor.py
Purpose: To process images before implementation in the Gaussian Naïve Bayes classifier
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"


# Import necessary libraries
import cv2
import numpy as np
from scipy import ndimage
import math


class ImageProcessor:
    def __init__(self, image_details) -> None:
        # image_details can be 1 of 2 things: a path to an image (from the code), or the actual image (from the website)

        # Initialize the image processor with the path to the image or image to be processed
        self.image_details = image_details

        # image_details is the actual image:
        if isinstance(image_details, np.ndarray):
            self.is_Image = True
        # image_details is a path to an image
        else:
            self.is_Image = False

    def process(self) -> np.ndarray:
        if self.is_Image == True:
            # Initialize image variable name with image array
            image = self.image_details
        elif self.is_Image == False:
            # Load the image with path in grayscale
            image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Resize and invert the image to a 20x20 pixel size for uniformity with training data
        image = cv2.resize(255 - image, (20, 20), interpolation=cv2.INTER_AREA)

        # Apply thresholding to convert the grayscale image to binary (black and white)
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove any empty (black) borders to center the digit
        image = self.trim_empty_borders(image)

        # Calculate the best shift to center the digit within the image
        shift_x, shift_y = self.calculate_best_shift(image)
        image = self.shift_image(image, shift_x, shift_y)

        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0

        # Reshape the image to a 1D array to match the model's input requirements
        image = image.reshape(-1)

        return image

    def trim_empty_borders(self, image) -> np.ndarray:
        # Remove empty borders around the digit to reduce noise and improve classification accuracy
        while np.sum(image[0]) == 0:
            image = image[1:]

        while np.sum(image[:, 0]) == 0:
            image = np.delete(image, 0, 1)

        while np.sum(image[-1]) == 0:
            image = image[:-1]

        while np.sum(image[:, -1]) == 0:
            image = np.delete(image, -1, 1)

        # Resize the image to maintain a standard size
        rows, cols = image.shape
        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
        image = cv2.resize(image, (cols, rows))

        # Add padding to ensure the image is 28x28 pixels
        col_padding = (
            int(math.ceil((28 - cols) / 2.0)),
            int(math.floor((28 - cols) / 2.0)),
        )
        row_padding = (
            int(math.ceil((28 - rows) / 2.0)),
            int(math.floor((28 - rows) / 2.0)),
        )
        image = np.pad(image, (row_padding, col_padding), "constant")

        return image

    def calculate_best_shift(self, image) -> np.int64:
        # Calculate the optimal shift to center the digit based on its center of mass
        center_y, center_x = ndimage.center_of_mass(image)
        rows, cols = image.shape
        shift_x = np.round(cols / 2.0 - center_x).astype(int)
        shift_y = np.round(rows / 2.0 - center_y).astype(int)

        return shift_x, shift_y

    def shift_image(self, image, shift_x, shift_y) -> np.ndarray:
        # Apply the calculated shift to the image to center the digit
        rows, cols = image.shape
        transformation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_image = cv2.warpAffine(image, transformation_matrix, (cols, rows))
        return shifted_image


def main() -> None:
    # Implementation of image processor in ./predict_digit_images.py
    pass


if __name__ == "__main__":
    main()
