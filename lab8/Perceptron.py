import numpy as np
import pandas as pd
from MnistLoader import MnistDataloader
import argparse
from matplotlib import pyplot as plt
import random


def show_images(images, title_texts):
    cols = 3
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0].reshape(28, 28)
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)  # type: ignore
        if title_text != "":
            plt.title(title_text, fontsize=15)
        index += 1

    plt.show()


class Perceptron:
    def __init__(self, n_features: int, hidden_layers, output_layer, lr=0.5) -> None:
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.lr = lr

        self.weights = [
            np.random.randn(hidden_layers[0], n_features) * np.sqrt(2 / n_features)
        ]  # HE initialization
        self.biases = [np.zeros((hidden_layers[0], 1))]
        for i in range(1, len(hidden_layers)):
            self.weights.append(
                np.random.randn(hidden_layers[i], hidden_layers[i - 1])
                * np.sqrt(2 / hidden_layers[i - 1])
            )
            self.biases.append(np.zeros((hidden_layers[i], 1)))

        self.weights.append(
            np.random.randn(output_layer, hidden_layers[-1])
            * np.sqrt(2 / hidden_layers[-1])
        )
        self.biases.append(np.zeros((output_layer, 1)))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_d(self, x):
        return np.where(x > 0, 1, 0)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_d(self, x):
        return x * (1 - x)

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_d(self, x):
        return 1 - np.tanh(x) ** 2

    def _softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    def _softmax_d(self, x):
        return x * (1 - x)

    def _activ_func(self, x, output_layer=False):
        if output_layer:
            return self._softmax(x)
        return self._tanh(x)

    def _activ_func_d(self, x):
        return self._tanh_d(x)

    def _loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def forward_prop(self, data: np.ndarray):
        self.activations = [data]
        self.zs = []  # input pre-activation

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            layer_input = np.dot(W, self.activations[-1]) + b
            self.zs.append(layer_input)
            layer_output = self._activ_func(
                layer_input, output_layer=(i == len(self.weights) - 1)
            )
            self.activations.append(layer_output)

        return self.activations[-1]

    def back_prop(self, y_true):
        m = self.activations[1].shape[1]  # Number of samples

        deltas = [self.activations[-1] - y_true]
        for i in range(len(self.hidden_layers), 0, -1):
            deltas.append(
                np.dot(self.weights[i].T, deltas[-1])
                * self._activ_func_d(self.zs[i - 1])
            )

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * np.dot(deltas[i], self.activations[i].T) / m
            self.biases[i] -= self.lr * np.sum(deltas[i], axis=1, keepdims=True) / m

    def train(self, x_train, y_train, epochs=100, batch_size=16, validation_data=None):
        m = x_train.shape[1]

        if validation_data:
            x_val, y_val = validation_data
            self.forward_prop(x_val)
            self.validation_loss = self._loss(y_val, self.activations[-1])

        for epoch in range(epochs):
            if epoch % 10 == 0:
                self.forward_prop(x_train)
                loss = self._loss(y_train, self.activations[-1])
                predictions = np.argmax(self.activations[-1], axis=0)
                print(f"Epoch: {epoch} Loss: {loss}")
                print(f"Accuracy: {np.mean(predictions == np.argmax(y_train, axis=0))}")

            if validation_data:
                self.forward_prop(x_val)
                validation_loss = self._loss(y_val, self.activations[-1])
                if validation_loss > 1.03 * self.validation_loss:
                    print("Early stopping")
                    break
                else:
                    self.validation_loss = validation_loss

            indices = np.random.permutation(m)
            x_train_shuffled = x_train[:, indices]
            y_train_shuffled = y_train[:, indices]

            # mini-batch gradient descent
            for i in range(0, m, batch_size):
                end = i + batch_size
                batch_x = x_train_shuffled[:, i:end]
                batch_y = y_train_shuffled[:, i:end]

                self.forward_prop(batch_x)
                self.back_prop(batch_y)


if __name__ == "__main__":
    mnist_dataloader = MnistDataloader()

    # data is already as np arrays, normalized and y data is in form like [0,0,0,1...] for number 3 etc.
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    digit_idx = [np.where(np.argmax(y_train, axis=1) == i)[0] for i in range(10)]
    test_digit_idx = [np.where(np.argmax(y_test, axis=1) == i)[0] for i in range(10)]
    for i in range(10):
        print(f"Number of {i}: {len(digit_idx[i])}")
        print(f"Indices of {i}: {digit_idx[i]}")

    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T

    print("----------------------------------")
    print("Data loaded")
    print("----------------------------------")

    network = Perceptron(
        n_features=784, hidden_layers=[16, 16], output_layer=10, lr=0.1
    )
    network.train(
        x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test.T)
    )

    results = (network.forward_prop(x_test)).T
    predictions = np.argmax(results, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    correct = 0
    for prediction, true_label in zip(predictions, true_labels):
        if prediction == true_label:
            correct += 1

    accuracy = correct / len(predictions)

    print(f"Accuracy: {accuracy}")
    print(x_train.shape)

    images = []
    titles = []
    wrong_predictions = np.where(predictions != true_labels)[0]
    for i in range(0, 6):
        images.append(x_test[:, test_digit_idx[7][i]])
        titles.append(
            "test image [" + str(i) + "] = " + str(predictions[test_digit_idx[7][i]])
        )

    # for i in range(0, 3):
    #     r = random.randint(1, 2000)
    #     images.append(x_test[:, r])
    #     titles.append("test image [" + str(r) + "] = " + str(predictions[r]))

    # for i in range(0, 3):  # display 3 random predicitions and 3 wrong ones
    #     w = random.randint(1, len(wrong_predictions))
    #     images.append(x_test[:, wrong_predictions[w]])
    #     titles.append("test image [" + str([w]) + "] = " + str(predictions[w]))

    show_images(images, titles)
