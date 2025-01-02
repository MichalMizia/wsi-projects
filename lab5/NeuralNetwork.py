import numpy as np
from sklearn.metrics import mean_squared_error


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        np.random.seed(1)
        # random weights from normal dostribution
        self.hidden_input_weights = np.random.randn(self.input_size, self.hidden_size)
        self.hidden_output_weights = np.random.randn(self.hidden_size, self.output_size)

    # can use sigmoid activation - if so targets (Y values) must be normalized to range (0, 1)
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def minMax_normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    # forward_prop with sigmoid activation
    def forward_prop(self, X):
        self.hidden_layer_input = np.dot(X, self.hidden_input_weights)
        self.hidden_layer_output = self._sigmoid(self.hidden_layer_input)

        self.final_layer_input = np.dot(
            self.hidden_layer_output, self.hidden_output_weights
        )
        self.final_layer_output = self._sigmoid(self.final_layer_input)

        return self.final_layer_output

    def back_prop(self, X, y, output):
        # output is in scope [0, 1]
        # y is normalized to be in same scope as output
        y = self.minMax_normalize(y)
        self.final_layer_error = y - output
        self.final_layer_delta = self.final_layer_error * self._sigmoid_derivative(
            output
        )

        self.hidden_layer_error = np.dot(
            self.final_layer_delta, self.hidden_output_weights.T
        )
        self.hidden_layer_delta = self.hidden_layer_error * self._sigmoid_derivative(
            self.hidden_layer_output
        )

        # delta = error * derivative
        # Weight += Input * delta * learning_rate --> gradient descent
        self.hidden_output_weights += (
            np.dot(self.hidden_layer_output.T, self.final_layer_delta) * self.lr
        )
        self.hidden_input_weights += np.dot(X.T, self.hidden_layer_delta) * self.lr

    def train(self, X, y, iters):
        self.loss = []
        for _ in range(iters):
            output = self.forward_prop(X)
            self.back_prop(X, y, output)
            self.loss.append(mean_squared_error(y, output))

    # def _relu(self, x):
    #     return np.maximum(0, x)

    # def _relu_derivative(self, x):
    #     return np.where(x > 0, 1, 0)

    # ReLU forward propagaton
    # def forward_prop(self, X):
    #     self.hidden_layer_input = np.dot(X, self.hidden_input_weights)
    #     self.hidden_layer_output = self._relu(
    #         self.hidden_layer_input
    #     )  # ReLU activation

    #     self.final_layer_input = np.dot(
    #         self.hidden_layer_output, self.hidden_output_weights
    #     )
    #     self.final_layer_output = self.final_layer_input  # linear activation

    #     return self.final_layer_output
