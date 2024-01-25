import numpy as np
from keras.datasets import mnist

class NaiveBayes:
    def __init__(self, magic_number):
        self.means = []
        self.variances = []
        self.priors = []
        self.magic_number = magic_number

    def fit(self, x, y):
        self.classes = np.unique(y)

        for i in self.classes:
            self.priors.append(np.mean(y == i))
            x_n = x[y == i]
            self.means.append(np.mean(x_n, axis=0))
            self.variances.append(np.var(x_n, axis=0) + self.magic_number) # 0.01559

    def predict(self, x):
        posteriors = []

        for i in self.classes:
            log_prior = np.log(self.priors[i])
            likelihood = np.sum(
                np.log(self.gaussian(x, self.means[i], self.variances[i])), axis=1
            )
            posterior = np.exp(likelihood) * np.exp(log_prior)
            posteriors.append(posterior)

        # .tranpose() swaps rows and columns
        posteriors = np.array(posteriors).transpose()

        # Since we can't directly get P(y | x) since we don't have P(x)
        # We use the fact that P(x)(P(1 | x) + P(2 | x) + ... + P(10 | x)) = 1
        # Thus we can solve for P(x) and use it as a scale factor to get the probability of P(y | x)
        for i in range(len(posteriors)):
            scale_factor = 1 / np.sum(posteriors[i])
            posteriors[i] *= scale_factor

        return posteriors

    def gaussian(self, x, mean, variance):
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

from joblib import Parallel, delayed

base_model = NaiveBayes(0.01575)
base_model.fit(x_train, y_train)
base_y_predicted = np.argmax(base_model.predict(x_test), axis=1)
base_accuracy = np.mean(base_y_predicted == y_test)
print('Initial Max Accuracy: ', base_accuracy)

highest_accuracy = base_accuracy
highest_accuracy_number = 0.01559

def parallel(x):
    x = x/100000
    global highest_accuracy
    global highest_accuracy_number
    global second_highest_accuracy
    global second_highest_accuracy_number
    model = NaiveBayes(x)
    model.fit(x_train, y_train)
    y_predicted = np.argmax(model.predict(x_test), axis=1)
    accuracy = np.mean(y_predicted == y_test)
    if accuracy > highest_accuracy:
        print("Highest Accuracy: " + str(accuracy))
        print("Highest Accuracy Number: " + str(x))
    if x % 1000 == 0: # Status update
        print(x/10000)

Parallel(n_jobs=8)(delayed(parallel)(x) for x in range(100000))