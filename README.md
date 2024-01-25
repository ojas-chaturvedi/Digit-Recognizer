# Handwritten Digit Recognizer

## Introduction

#### In this project, we have developed a Gaussian Naïve Bayes classifier from scratch to classify the MNIST dataset, which consists of 70,000 28x28 grayscale images of handwritten digits. We have also created a website where users can easily try out the Python model as our deliverable.

#### Our model is based on Bayesian statistics, which views probabilities as uncertain values that can be altered when new evidence is revealed (this is where the idea of conditional probabilities originates from). This differs from Frequentist statistics, which conversely views probabilities as fixed values based on the long-run relative frequency of an event occurring over repeated trials. The formula for Bayes Theorem can be seen below:

![Bayes' Theorem Formula](/writing/Bayes.jpg)

#### * P(Y | X)/Posterior probability: The updated probability of your belief occurring based on how likely the event was to occur before new evidence was introduced (the prior probability) and how likely this evidence was to appear for the given class (the likelihood)
#### * P(X | Y)/Likelihood: The probability of the evidence/features appearing given a certain belief/class
#### * P(Y)/Prior probability: How likely the initial belief/event was to occur without any evidence
#### * P(X)/Normalization constant. Probability of evidence

#### Note: Our model is called “Naïve” because it is assumed that the data’s features are independent of each other for any given class label.

#### For reference, here is an example of a number from the MNIST dataset:

![Example of MNIST number](/writing/MNIST_ex.png)
