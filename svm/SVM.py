import numpy as np


class SVM:
    def __init__(self, kernel=None, degree=3, learning_rate=0.001, gamma=0.1) -> None:
        self.lr = learning_rate

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma

        self.weights = None
        self.bias = None

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def polynomial_kernel(self, X1, X2):
        return (1 + np.dot(X1, X2.T)) ** self.degree

    def rbf_kernel(self, X1, X2):
        if self.gamma is None:
            self.gamma = 1 / X1.shape[1]
        return np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)

    def compute_kernel(self, X1, X2):
        if self.kernel == "linear":
            return self.linear_kernel(X1, X2)
        elif self.kernel == "polynomial":
            return self.polynomial_kernel(X1, X2)
        elif self.kernel == "rbf":
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y, iters=1000):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.initial = X.copy()

        print("X samples and features: ", n_samples, n_features)

        self.weights = np.random.rand(n_features)
        self.bias = 0

        if self.kernel:
            X = self.compute_kernel(X, X)
            self.weights = np.zeros(X.shape[0])

        for i in range(iters):
            for y_value, X_value in zip(y_, X):
                estimate = y_value * (np.dot(X_value, self.weights) - self.bias)
                if estimate >= 1:
                    self.weights -= self.lr * self.weights
                    # decrease the weights to make the decision boundary as large as possible
                else:  # the point is on the wrong side of the decision boundary
                    self.weights -= self.lr * (
                        (1 - estimate) * self.weights - y_value * X_value
                    )
                    self.bias -= self.lr * y_value

    def predict(self, X):
        if self.weights is None:
            return np.empty(0)

        if self.kernel:
            X = self.compute_kernel(X, self.initial)

        print(self.weights.shape, X.shape)

        return np.array([np.dot(X_value, self.weights) - self.bias for X_value in X])
