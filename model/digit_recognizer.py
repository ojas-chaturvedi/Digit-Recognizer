import numpy as np
import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import os
import math
from scipy import ndimage

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

class ProcessImage:
    def __init__(self, image_path):
        self.path = image_path

    def preprocess(self):
        # Read the image
        img = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        # Scale to 20x20, invert (like training)
        img = cv2.resize(255 - img, (20, 20), interpolation = cv2.INTER_AREA)

        # img = cv2.GaussianBlur(img,(5,5),0)

        # Make gray into black (uniform background like training)
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove completely black (empty) rows/cols on all sides
        img = self.trim(img)

        colsPadding = (int(math.ceil((28 - cols)/ 2.0)), int(math.floor((28 - cols)/ 2.0)))
        rowsPadding = (int(math.ceil((28 - rows)/ 2.0)), int(math.floor((28 - rows)/ 2.0)))
        img = np.pad(img, (rowsPadding, colsPadding), 'constant')

        # Center digit
        shiftx, shifty = self.getBestShift(img)
        shifted = self.shift(img,shiftx,shifty)
        img = shifted

        # DEBUG
        cv2.imwrite("output.png", img)

        # Normalize the image
        img = img / 255.0

        # Reshape to 1D match the input of the model
        img = img.reshape(-1)
        return img

    def trim(self, img):
        while np.sum(img[0]) == 0:
            img = img[1:]

        while np.sum(img[:,0]) == 0:
            img = np.delete(img,0,1)

        while np.sum(img[-1]) == 0:
            img = img[:-1]

        while np.sum(img[:,-1]) == 0:
            img = np.delete(img,-1,1)

        rows,cols = img.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            img = cv2.resize(img, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            img = cv2.resize(img, (cols, rows))


        return img

    def getBestShift(self, img):
        cy, cx = ndimage.center_of_mass(img)

        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty

    def shift(self, img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
# x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
#
model = NaiveBayes()
model.fit(x_train, y_train)
# y_predicted = model.predict(x_test)
# accuracy = np.mean(y_predicted == y_test)
# print("Accuracy: ", accuracy)

digits_imgs = []
for file in os.scandir("digits"):
    img_processor = ProcessImage(file.path)
    img = img_processor.preprocess()
    digits_imgs.append(img)

predicted_digits = model.predict(digits_imgs)
print("Predicted Digit: ", predicted_digits)

actual_digits = [7, 7, 0, 5, 3, 2, 1, 0, 8, 7, 4, 2, 9, 8, 5, 1, 1, 1, 7]
print("Actual Digits: ", actual_digits)

accuracy = np.mean(predicted_digits == actual_digits)
print("Accuracy: ", accuracy)

# img_processor = ProcessImage("testing.png")
# img = img_processor.preprocess()
# predicted_digit = model.predict([img])
# print("Predicted Digit: ", predicted_digit)
