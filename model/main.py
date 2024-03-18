#!/opt/homebrew/bin/python3
"""
Name: main.py
Purpose: To implement a Gaussian Naïve Bayes classifier to recognize handwritten digits
"""

__author__ = "Ojas Chaturvedi"
__github__ = "github.com/ojas-chaturvedi"
__license__ = "MIT"


# Import necessary libraries
import numpy as np
from keras.datasets import mnist

# Import project code
from NaiveBayesClassifier import NaiveBayesClassifier


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


def main() -> None:
    print(f"Accuracy of model: {accuracy}")

    # Implementation of testing on online and self-made images in ./predict_digit_images.py


if __name__ == "__main__":
    main()
