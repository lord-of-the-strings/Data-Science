import numpy as np


class LogisticRegression(object):
    """Logistic Regression blueprint"""

    def __init__(self, alpha, epoch):
        """Constructor to intialize basic math and gradient descent properties"""
        self.alpha = alpha
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-z))

    def _cost(self, h, y):
        """Cross entropy cost function"""
        m = len(y)
        return -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        """Runs gradient descent to opytimize _cost and predict the line of separation"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        for i in range(self.epoch):
            print(f"Running epoch [{i + 1}/{self.epoch}................]")
            z = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(z)
            dw = 1 / m * np.dot(X.T, h - y)
            db = 1 / m * np.sum(h - y)
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        """Binary classification"""
        return (self._sigmoid(np.dot(X, self.weights) + self.bias) >= 0.5).astype(int)  # pyright: ignore
