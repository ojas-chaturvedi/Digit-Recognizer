#!/opt/homebrew/bin/python3
"""
Name: digit_recognizer.py
Purpose: To implement a Gaussian Naïve Bayes classifier to recognize handwritten digits
"""

__author__ = "Ojas Chaturvedi"
__github__ = "ojas-chaturvedi"
__license__ = "MIT"


# Import necessary libraries
import numpy as np
from keras.datasets import mnist
import cv2
import math
from scipy import ndimage


class NaiveBayesClassifier:
    def __init__(self: object) -> None:
        # Initialize lists to store means, variances, and priors for each class
        self.class_means = []
        self.class_variances = []
        self.class_priors = []

    def fit(self: object, features: np.ndarray, labels: np.ndarray) -> None:
        # Train the classifier with features and labels
        self.unique_classes = np.unique(labels)

        for class_index in self.unique_classes:
            self.class_priors.append(np.mean(labels == class_index))
            features_for_class = features[labels == class_index]
            self.class_means.append(np.mean(features_for_class, axis=0))
            self.class_variances.append(np.var(features_for_class, axis=0) + 0.01575)

    def predict(self: object, features: np.ndarray) -> np.ndarray:
        # Predict the class for each feature set in features
        class_posteriors = []

        for class_index in self.unique_classes:
            log_prior = np.log(self.class_priors[class_index])
            class_likelihood = np.sum(
                np.log(
                    self.gaussian_distribution(
                        features,
                        self.class_means[class_index],
                        self.class_variances[class_index],
                    )
                ),
                axis=1,
            )
            posterior = np.exp(class_likelihood) * np.exp(log_prior)
            class_posteriors.append(posterior)

        class_posteriors = np.array(class_posteriors).transpose()

        # Normalize the posteriors to get probabilities
        for i in range(len(class_posteriors)):
            normalization_factor = 1 / np.sum(class_posteriors[i])
            class_posteriors[i] *= normalization_factor

        return class_posteriors

    def gaussian_distribution(
        self: object, x: np.ndarray, mean: np.ndarray, variance: np.ndarray
    ) -> np.ndarray:
        # Calculate the Gaussian distribution
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)

        return numerator / denominator


class ImagePreprocessor:
    def __init__(self: object, image_path) -> None:
        # Initialize the image processor with the path to the image to be processed
        self.image_path = image_path

    def preprocess(self: object) -> np.ndarray:
        # Load the image in grayscale
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

    def trim_empty_borders(self: object, image) -> np.ndarray:
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

    def calculate_best_shift(self: object, image) -> np.int64:
        # Calculate the optimal shift to center the digit based on its center of mass
        center_y, center_x = ndimage.center_of_mass(image)
        rows, cols = image.shape
        shift_x = np.round(cols / 2.0 - center_x).astype(int)
        shift_y = np.round(rows / 2.0 - center_y).astype(int)

        return shift_x, shift_y

    def shift_image(self: object, image, shift_x, shift_y) -> np.ndarray:
        # Apply the calculated shift to the image to center the digit
        rows, cols = image.shape
        transformation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_image = cv2.warpAffine(image, transformation_matrix, (cols, rows))
        return shifted_image


# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Train the Naive Bayes classifier
model = NaiveBayesClassifier()
model.fit(x_train, y_train)

# Evaluate the classifier
y_pred = np.argmax(model.predict(x_test), axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")


def predict_digit_images_one() -> None:
    # Initialize a list to hold preprocessed digit images
    image_suffixes = [
        "1",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    preprocessed_images = []

    # Load and preprocess each image
    for suffix in image_suffixes:
        image_processor = ImagePreprocessor(
            f"model/testing_images/set_one/image{suffix}.png"
        )
        preprocessed_image = image_processor.preprocess()
        preprocessed_images.append(preprocessed_image)

    # Predict the digit for each preprocessed image
    predicted_labels = np.argmax(model.predict(preprocessed_images), axis=1)
    print(f"Predicted Digits: {predicted_labels}")

    # Actual labels for the digits
    true_labels = np.array([7, 7, 0, 5, 3, 2, 1, 0, 8, 7, 4, 2, 9, 8, 5, 1, 1, 1, 7])
    print(f"Actual Digits: {true_labels}")

    # Calculate and print the accuracy of the predictions
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Prediction Accuracy: {accuracy}")


def predict_digit_images_two() -> None:
    # Initialize a list to hold preprocessed digit images
    image_suffixes = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "11",
        "22",
        "44",
        "88",
        "99",
        "111",
    ]
    preprocessed_images = []

    # Load and preprocess each image
    for suffix in image_suffixes:
        image_processor = ImagePreprocessor(
            f"model/testing_images/set_two/image{suffix}.png"
        )
        preprocessed_image = image_processor.preprocess()
        preprocessed_images.append(preprocessed_image)

    # Predict the digit for each preprocessed image
    predicted_labels = np.argmax(model.predict(preprocessed_images), axis=1)
    print(f"Predicted Digits: {predicted_labels}")

    # Actual labels for the digits
    true_labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 4, 8, 9, 1])
    print(f"Actual Digits: {true_labels}")

    # Calculate and print the accuracy of the predictions
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Prediction Accuracy: {accuracy}")


def main() -> None:
    # Execute the functions to predict set one and two digit images
    predict_digit_images_one()
    predict_digit_images_two()


if __name__ == "__main__":
    main()
