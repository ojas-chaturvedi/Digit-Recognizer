#!/opt/homebrew/bin/python3
"""
Name: NaiveBayesClassifier.py
Purpose: To create a Gaussian Naïve Bayes classifier to recognize handwritten digits
"""

__author__ = "Ojas Chaturvedi"
__github__ = "ojas-chaturvedi"
__license__ = "MIT"


import numpy as np


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


def main() -> None:
    # TODO Create model implementation
    pass


if __name__ == "__main__":
    main()
