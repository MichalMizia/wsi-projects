import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, learn_rate=0.05, iters=1000) -> None:
        self.learn_rate = learn_rate
        self.iters = iters
        self.weights = None
        self.bias = None
        self.losses = []

    def calculate_loss(self, y_pred, y_true):
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def fit(self, X, y):
        samples, sample_features = X.shape
        self.weights = np.zeros(sample_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.iters):
            linear = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear)

            self.losses.append(float(self.calculate_loss(predictions, y)))

            d_weights = (1 / samples) * np.dot(X.T, (predictions - y))
            d_bias = (1 / samples) * np.sum((predictions - y))

            self.weights -= d_weights * self.learn_rate
            self.bias -= d_bias * self.learn_rate

        print("Weights: ", self.weights)
        print("Losses: ", self.losses)

    def predict(self, X):
        if self.weights is None or not self.bias:
            return []

        linear = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear)
        return [0 if y < 0.5 else 1 for y in y_pred]
