import numpy as np


def relu(x):
    # ReLU is a rectified linear unit function that returns a value between 0 and x.
    # relu = 0 if x < 0 else x
    return max(0, x)


def sigmoid(x):
    # Sigmoid is a sigmoid function that returns a value between 0 and 1.
    # sigmoid = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def tanh(x):
    # Tanh is a hyperbolic tangent function that returns a value between -1 and 1.
    # tanh = (e^x - e^(-x)) / (e^x + e^(-x))
    return np.tanh(x)


def softmax(x):
    # Softmax is a softmax function that returns a value between 0 and 1.
    # softmax = e^x / sum(e^x)
    return np.exp(x) / np.sum(np.exp(x))
