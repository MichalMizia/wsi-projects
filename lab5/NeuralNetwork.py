import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        np.random.seed(1)
        self.hidden_input_weights = np.random.randn(
            self.input_size, self.hidden_size
        )  # random weights from normal dostribution
        self.hidden_output_weights = np.random.randn(self.hidden_size, self.output_size)

    # can use sigmoid activation - if so targets (Y values) must be normalized to range (0, 1)
    # def _sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def _sigmoid_derivative(self, x):
    #     return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward_prop(self, X):
        self.hidden_layer_input = np.dot(X, self.hidden_input_weights)
        self.hidden_layer_output = self._relu(
            self.hidden_layer_input
        )  # ReLU activation

        self.final_layer_input = np.dot(
            self.hidden_layer_output, self.hidden_output_weights
        )
        self.final_layer_output = self.final_layer_input  # linear activation

        return self.final_layer_output

    def back_prop(self, X, y, output):
        self.final_layer_error = y - output
        self.final_layer_delta = (
            self.final_layer_error * 1
        )  # linear function derivatove is 1

        self.hidden_layer_error = np.dot(self.hidden_output_weights.T, self.final_layer_delta)
        self.hidden_layer_delta = self.hidden_error * self._relu_derivative(self.hidden_layer_output)

        # Weight += Error * Input * derrivative, error * derivative = delta
        self.hidden_output_weights += (
            np.dot(self.hidden_layer_output.T, self.final_layer_delta) * self.lr
        )
        self.hidden_input_weights += np.dot(X.T, self.hidden_layer_delta) * self.lr

    def train(self):
        pass
