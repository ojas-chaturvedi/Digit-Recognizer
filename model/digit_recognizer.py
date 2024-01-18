import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import os

class NaiveBayes:
    def __init__(self):
        self.means = []
        self.variances = []
        self.priors = []
        self.posteriors = []

    def fit(self, x, y):
        self.classes = np.unique(y)
        hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)

        for i in self.classes:
            self.priors.append(np.mean(y == i))

            x_i = x[y == i]
            x_hog = np.zeros((len(x_i), 324))
            for j in range(len(x_i)):
                x_hog[j] = np.squeeze(hog.compute(x_i[j].reshape(28, 28)))

            self.means.append(np.mean(x_hog, axis = 0))
            self.variances.append(np.var(x_hog, axis = 0) + 0.01559)

    def predict(self, x):
        self.posteriors = []
        hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)
        x_hog = np.zeros((len(x), 324))
        for i in self.classes:
            for j in range(len(x)):
                x_hog[j] = np.squeeze(hog.compute(x[j].reshape(28, 28)))

            log_prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self.gaussian(x_hog, self.means[i], self.variances[i])), axis = 1)
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

class ProcessImage:
    def __init__(self):
        self.path = 'random.PNG'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

model = NaiveBayes()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
accuracy = np.mean(y_predicted == y_test)
print("Accuracy: ", accuracy)
