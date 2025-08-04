import numpy as np


class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_predictions = np.dot(X, self.weights) + self.bias
            self.cost = (1 / n_samples) * np.sum((linear_predictions - y) ** 2)
            d_weights = (1 / n_samples) * np.dot(X.T, (linear_predictions - y))
            d_bias = (1 / n_samples) * np.sum(linear_predictions - y)

            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        # y = weights * X + bias
        return np.dot(X, self.weights) + self.bias

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_bias(self) -> float:
        return self.bias

    def get_cost(self) -> float:
        return self.cost
