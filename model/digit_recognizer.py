import numpy as np
from keras.datasets import mnist

# Bayes theorem: P(y|x) = P(x|y) * P(y) / P(x)
    # P(y|x) is the posterior
    # P(y) is the prior
        # Frequency of each class
    # P(x|y) is the likelihood
        # = P(x_1|y) * P(x_2|y) * ... * P(x_n|y)
            # P(x_n|y) is a feature
# Step 1: Load the data
# Step 2: Preprocess the data using reshape
# fit() idk why this is called fit, just using convention
    # Step 3: Find prior of each class (0-9) by looping through np.unique(y) and store in an array
    # Step 4: Find mean and variance for each feature and store in an array
# predict() - finding the class with the highest posterior (calculate y)
    # y = argmax(P(y|x)) = argmax(P(x|y) * P(y) / P(x)) = argmax(P(x|y) * P(y)) = argmax(log(P(x_1|y) * P(x_2|y) * ... * P(x_n|y) * P(y)))

class NaiveBayes:
    def __init__(self):
        self.means = []
        self.variances = []
        self.priors = []
        self.posteriors = []

    def fit(self, x, y):
        self.classes = np.unique(y)

        for i in self.classes:
            self.priors.append(np.sum(y == i) / len(y))
            x_n = x[y == i]
            self.means.append(np.mean(x_n, axis = 1))
            self.variances.append(np.var(x_n, axis = 1))

    def predict(self, x):
        for i in self.classes:
            log_prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(1 / np.sqrt(2 * np.pi * self.variances[i])) - ((x - self.means[i]) ** 2) / (2 * self.variances[i]))
            posterior = likelihood + log_prior
            self.posteriors.append(posterior)
            y = np.argmax(self.posteriors)
        
        return y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

model = NaiveBayes()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
accuracy = np.mean(y_predicted == y_test)
print(accuracy)