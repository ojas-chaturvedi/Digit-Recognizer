# Handwritten Digit Recognizer

## Introduction

#### In this project, we have developed a Gaussian Naïve Bayes classifier from scratch to classify the MNIST dataset, which consists of 70,000 28x28 grayscale images of handwritten digits. We have also created a website where users can easily try out the Python model.

#### Our model is based on Bayesian statistics, which views probabilities as uncertain values that can be altered when new evidence is revealed (this is where the idea of conditional probabilities originates from). This differs from Frequentist statistics, which conversely views probabilities as fixed values based on the long-run relative frequency of an event occurring over repeated trials. The formula for Bayes Theorem can be seen below:

![Bayes' Theorem Formula](/writing/Bayes.jpg)

* P(Y | X)/Posterior probability: The updated probability of your belief occurring based on how likely the event was to occur before new evidence was introduced (the prior probability) and how likely this evidence was to appear for the given class (the likelihood)
* P(X | Y)/Likelihood: The probability of the evidence/features appearing given a certain belief/class
* P(Y)/Prior probability: How likely the initial belief/event was to occur without any evidence
* P(X)/Normalization constant. Probability of the evidence

#### Note: Our model is called “Naïve” because it is assumed that the data’s features are independent of each other for any given class label.

#### For reference, here is an example of a number from the MNIST dataset:

![Example of MNIST Number](/writing/MNIST_ex.png)

## Model Implementation

#### The following section will run through our code line-by-line for our model:

1. This block of code imports the MNIST dataset and reshapes the feature arrays from 3D to 2D to make these features easier to work with.
```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0
```
2. This code defines our model as an object, with three attributes (means, variances, priors) that will be used later when performing classification.
```python
class NaiveBayes:
  def __init__(self):
  self.means = []
  self.variances = []
  self.priors = []

  def fit(self, x, y):
    self.classes = np.unique(y)

    for i in self.classes:
      self.priors.append(np.mean(y == i))
      x_i = x[y == i]
      self.means.append(np.mean(x_i, axis = 0))
      self.variances.append(np.var(x_i, axis = 0) + 0.01575)
```
3. This code saves the ten classes from the dataset in the “classes” attribute. Additionally, for each class, the model saves the mean and variance for each pixel. This will be used later when we use a Gaussian to help classify our data (a Gaussian is used because our features are continuous, not discrete). The hard coded parameter 0.01575 exists to ensure no variance value is 0 (this would cause a divide by 0 error when implementing the Gaussian).
```python
  def predict(self, x):
    posteriors = []

    for i in self.classes:
      log_prior = np.log(self.priors[i])
      likelihood = np.sum(np.log(self.gaussian(x, self.means[i], self.variances[i])), axis = 1)
      posterior = likelihood + log_prior
      posteriors.append(posterior)
```
4. This code defines the Gaussian.
```python
  def gaussian(self, x, mean, variance):
    numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
    denominator = np.sqrt(2 * np.pi * variance)
        
    return numerator / denominator
```
5. This code computes the log of the priors, the log of the p-value for each pixel in a given training/testing example, and the posteriors of each class. Then, the class corresponding with the maximum posterior is returned.
- Note: The posteriors variable doesn’t hold the actual posteriors of each class because we summed the priors with the likelihoods.
```python
  def predict(self, x):
    posteriors = []

    for i in self.classes:
      log_prior = np.log(self.priors[i])
      likelihood = np.sum(np.log(self.gaussian(x, self.means[i], self.variances[i])), axis = 1)
      posterior = likelihood + log_prior
      posteriors.append(posterior)

    return np.argmax(posteriors, axis = 0)
```
6. This code classifies the test examples of the dataset and returns the accuracy score of the model.
```python
model = NaiveBayes()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
accuracy = np.mean(y_predicted == y_test)
print("Accuracy: ", accuracy)
```

#### For our website, we aimed to have users submit their pictures of digits on white backgrounds, so people can see how our model classified their image. However, we needed to implement a preprocessing pipeline for these images to be in the same format as the MNIST examples our model was trained on. Thus, this code block uses functions to scale the image, invert the image’s colors, center the number, and pad the image.

## Discussion/Limitations

#### Our model, although having an accuracy of 81.56%, still has some limitations in accuracy, as seen by our confusion matrix. For example, from the heat map below, it can be seen that our model often mistakes 4s for 9s. This could be because of how similar the heat maps of these two classes are, but we are unsure exactly why this happens.

#### Confusion Matrix

![Confusion Matrix](/writing/ConfusionMatrix.png)

#### Heat Map of a 4 and 9, respectively

![Heat Map of a 4](/writing/HeatMap4.png)

![Heat Map of a 9](/writing/HeatMap9.png)

#### Example of a misclassification

![Misclassification Example](/writing/Misclassification.jpg)

#### Additionally, when we experimented with the posterior probabilities, we saw that our model was always 100% sure of its predicted number, even if its prediction was wrong. Again, we are not sure why this occurs.

## [Here is Our Website!](https://handwritten-digit-recognizer.streamlit.app/)
https://handwritten-digit-recognizer.streamlit.app/