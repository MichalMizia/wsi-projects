import numpy as np
import pandas as pd
from MnistLoader import MnistDataloader
import argparse
import matplotlib.pyplot as plt
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
    def __init__(self, n_features: int = 784, lr=0.1) -> None:
        self.lr = lr

        self.W1 = np.random.randn(16, 784) * np.sqrt(2 / n_features)
        self.W2 = np.random.randn(10, 16) * np.sqrt(2 / 16)
        self.B1 = np.zeros((16, 1))
        self.B2 = np.zeros((10, 1))

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_d(self, x):
        return np.where(x > 0, 1, 0)

    def _softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def _loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def _loss_d(self, y_true, y_pred):
        return y_pred - y_true

    def forward_prop(self, data: np.ndarray):
        # data: (784, m), W1: (784, 16)
        self.Z1 = np.dot(self.W1, data) + self.B1  # Z1: (16, m)
        self.A1 = self._relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.B2  # W2: (16, 10), Z2: (10, m)
        self.A2 = self._softmax(self.Z2)

        return self.A2

    def back_prop(self, data: np.ndarray, y_true):
        m = data.shape[1]  # Number of samples

        dZ2 = self.A2 - y_true  # (10, m)
        dW2 = np.dot(dZ2, self.A1.T) / m  # (10, 16)
        dB2 = np.sum(dZ2, axis=1, keepdims=True) / m  # (10, 1)

        dZ1 = np.dot(self.W2.T, dZ2) * self._relu_d(self.Z1)  # (16, m)
        dW1 = np.dot(dZ1, data.T) / m  # (16, 784)
        dB1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (16, 1)

        self.W2 -= self.lr * dW2
        self.B2 -= self.lr * dB2
        self.W1 -= self.lr * dW1
        self.B1 -= self.lr * dB1

    def train(self, x_train, y_train, epochs=100, batch_size=32):
        m = x_train.shape[1]

        for epoch in range(epochs):
            if epoch % 10 == 0:
                self.forward_prop(x_train)
                loss = self._loss(y_train, self.A2)
                predictions = np.argmax(self.A2, axis=0)
                print(f"Epoch: {epoch} Loss: {loss}")
                print(f"Accuracy: {np.mean(predictions == np.argmax(y_train, axis=0))}")

            indices = np.random.permutation(m)
            x_train_shuffled = x_train[:, indices]
            y_train_shuffled = y_train[:, indices]

            # mini-batch gradient descent
            for i in range(0, m, batch_size):
                end = i + batch_size
                batch_x = x_train_shuffled[:, i:end]
                batch_y = y_train_shuffled[:, i:end]

                # print(batch_x.shape, batch_y.shape)  # (784, 32), (10, 32)

                self.forward_prop(batch_x)
                self.back_prop(batch_x, batch_y)


if __name__ == "__main__":
    mnist_dataloader = MnistDataloader()

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()  # as np arrays

    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.T

    print("----------------------------------")
    print("Data loaded")
    print("----------------------------------")

    network = Perceptron(784, lr=0.5)
    network.train(x_train, y_train, epochs=40)

    results = (network.forward_prop(x_test)).T
    predictions = np.argmax(results, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    correct = 0
    for prediction, true_label in zip(predictions, true_labels):
        if prediction == true_label:
            correct += 1

    accuracy = correct / len(predictions)

    images = []
    titles = []
    wrong_predictions = np.where(predictions != true_labels)[0]
    print("Wrong predictions: ", wrong_predictions)
    for i in range(0, 3):
        r = random.randint(1, 2000)
        images.append(x_test[:, r])
        titles.append("training image [" + str(r) + "] = " + str(predictions[r]))

    for i in range(0, 3):
        w = random.randint(1, len(wrong_predictions))
        images.append(x_test[:, wrong_predictions[w]])
        titles.append("training image [" + str([w]) + "] = " + str(predictions[w]))

    show_images(images, titles)

    print("Results: ", results[0:10])
    print(f"Accuracy: {accuracy}")
