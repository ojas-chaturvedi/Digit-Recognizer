import numpy as np
from keras.datasets import mnist
import os, cv2, numpy as np, tensorflow as tf, matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        self.means = []
        self.variances = []
        self.priors = []
        self.posteriors = []

    def fit(self, x, y):
        self.classes = np.unique(y)

        for i in self.classes:
            self.priors.append(np.mean(y == i))
            x_n = x[y == i]
            self.means.append(np.mean(x_n, axis = 0))
            self.variances.append(np.var(x_n, axis = 0) + 0.01559)

    def predict(self, x):
        self.posteriors = []
        for i in self.classes:
            log_prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self.gaussian(x, self.means[i], self.variances[i])), axis = 1)
            posterior = likelihood + log_prior
            self.posteriors.append(posterior)
        self.posteriors = np.array(self.posteriors)
        if self.posteriors.ndim == 2:
            return np.argmax(self.posteriors, axis=0)
        else:
            return np.argmax(self.posteriors)

    def gaussian(self, x, mean, variance):
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
        
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

model = NaiveBayes()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
accuracy = np.mean(y_predicted == y_test)
print("Accuracy: ", accuracy)