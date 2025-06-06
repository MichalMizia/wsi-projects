import numpy as np
from sklearn.metrics import mean_squared_error


class NeuralNetwork:
    def __init__(
        self, input_size, hidden_layers_size, output_size, lr, weights_init="RDN"
    ):
        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.output_size = output_size
        self.lr = lr
        self.weights_init = weights_init

        # np.random.seed(1)

        # He weights initalization (randm - vlues from normal dostribution)
        # weights of input layer
        self.weights = [
            (
                np.random.randn(self.input_size, self.hidden_layers_size[0])
                * np.sqrt(2 / self.input_size)
                if self.weights_init == "HE"
                else np.random.uniform(
                    -0.1, 0.1, (self.input_size, self.hidden_layers_size[0])
                )
            )
        ]

        # weights of hidden layers
        for i in range(1, len(self.hidden_layers_size)):
            self.weights.append(
                np.random.randn(
                    self.hidden_layers_size[i - 1], self.hidden_layers_size[i]
                )
                * np.sqrt(2 / self.hidden_layers_size[i - 1])
                if self.weights_init == "HE"
                else np.random.uniform(
                    -0.1,
                    0.1,
                    (self.hidden_layers_size[i - 1], self.hidden_layers_size[i]),
                )
            )

        # weights of output layers
        self.weights.append(
            np.random.randn(self.hidden_layers_size[-1], self.output_size)
            * np.sqrt(2 / self.hidden_layers_size[-1])
            if self.weights_init == "HE"
            else np.random.uniform(
                -0.1, 0.1, (self.hidden_layers_size[-1], self.output_size)
            )
        )

        self.biases = [np.random.randn(1, size) for size in self.hidden_layers_size]
        self.biases.append(np.random.randn(1, self.output_size))

    # common activation functions - sigmoid, ReLU, tanh
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _ReLU(self, x):
        return np.maximum(0, x)

    def _ReLU_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _leaky_ReLU(self, x):
        alpha = -0.01
        return np.where(x > 0, x, alpha * x)

    def _leaky_ReLU_derivative(self, x):
        alpha = 0.01
        return np.where(x > 0, 1, alpha)

    def _activ_func(self, x):
        # return self._sigmoid(x)
        # return self._ReLU(x)
        return self._leaky_ReLU(x)

    def _activ_func_derivative(self, x):
        # return self._sigmoid_derivative(x)
        # return self._ReLU_derivative(x)
        return self._leaky_ReLU_derivative(x)

    def _mse_derivative(self, y, output):
        return 2 * (output - y) / y.size

    # forward_prop with relu activation
    def forward_prop(self, X):
        self.layers_inputs = []
        self.layers_outputs = [X]
        for i in range(len(self.weights) - 1):
            layer_input = (
                np.dot(self.layers_outputs[-1], self.weights[i]) + self.biases[i]
            )
            self.layers_inputs.append(layer_input)
            layer_output = self._activ_func(layer_input)
            self.layers_outputs.append(layer_output)

        final_layer_output = (
            np.dot(self.layers_outputs[-1], self.weights[-1]) + self.biases[-1]
        )
        return final_layer_output

    def back_prop(self, X, y, output):

        # dLoss / dWeights = (dLoss / dOutput) * (dOutput / dInput) * ... * (dInput / dWeights)]
        # loss is MSE - (1/N) * sum((y - output)**2)

        first_dLoss_dInput = self._mse_derivative(
            y, output
        ) * self._activ_func_derivative(output)
        # dLoss_dInput = (dLoss / dOutput) * (dOutput / dInput)
        layers_dLoss_dInput = [first_dLoss_dInput]

        for i in range(len(self.hidden_layers_size) - 1, -1, -1):
            dLoss_dInput = np.dot(
                layers_dLoss_dInput[-1], self.weights[i + 1].T
            ) * self._activ_func_derivative(self.layers_inputs[i])
            # self.weights[i + 1] cause' for n hidden layers there's always n+1 weight sets
            layers_dLoss_dInput.append(dLoss_dInput)

        layers_dLoss_dInput.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= (
                np.dot(self.layers_outputs[i].T, layers_dLoss_dInput[i]) * self.lr
            )
            self.biases[i] -= (
                np.expand_dims(np.sum(layers_dLoss_dInput[i], axis=0), axis=0) * self.lr
            )

    def train(self, X, y, iters):
        self.loss = []
        for _ in range(iters):
            output = self.forward_prop(X)
            self.back_prop(X, y, output)
            self.loss.append(mean_squared_error(y, output))
