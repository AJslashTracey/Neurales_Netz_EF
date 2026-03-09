import numpy as np
import matplotlib.pyplot as plt


class DigitalNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    