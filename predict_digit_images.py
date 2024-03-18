#!/opt/homebrew/bin/python3
"""
Name: predict_digit_images.py
Purpose: To test the Gaussian Naïve Bayes classifier on online and self-made images
"""

__author__ = "Ojas Chaturvedi"
__github__ = "ojas-chaturvedi"
__license__ = "MIT"


# Import necessary libraries
import numpy as np

# Import project code
from ImageProcessor import ImageProcessor
from main import model


def set_one() -> None:
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
        image_processor = ImageProcessor(
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


def set_two() -> None:
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
        image_processor = ImageProcessor(
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
    # TODO Create implementation of testing
    pass


if __name__ == "__main__":
    main()