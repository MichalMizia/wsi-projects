import numpy as np
import struct
from array import array
from os.path import join
import pandas as pd


class MnistDataloader(object):
    def __init__(
        self,
        training_images_filepath="data/train-images-idx3-ubyte/train-images-idx3-ubyte",
        training_labels_filepath="data/train-labels-idx1-ubyte/train-labels-idx1-ubyte",
        test_images_filepath="data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
        test_labels_filepath="data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
    ):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    "Magic number mismatch, expected 2049, got {}".format(magic)
                )
            labels = array("B", file.read())

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    "Magic number mismatch, expected 2051, got {}".format(magic)
                )
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )

        x_train, y_train, x_test, y_test = (
            np.array(x_train),
            np.array(y_train),
            np.array(x_test),
            np.array(y_test),
        )

        x_train = x_train / 255.0
        x_test = x_test / 255.0  # normalize

        # convert number like '5' to list [0,0,0,0,0,1,....]
        y_train = pd.get_dummies(y_train, dtype=int).values
        y_test = pd.get_dummies(y_test, dtype=int).values

        x_train = x_train.reshape(x_train.shape[0], -1)  # reshape to 1D, 28*28 = 784
        x_test = x_test.reshape(x_test.shape[0], -1)

        return (x_train, y_train), (x_test, y_test)
