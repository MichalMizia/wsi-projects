# Custom MLP built using only numpy

## Running

```bash
python -m venv env
./env/Scripts/activate
pip install -r requirements.txt

python ./board.py
```

https://github.com/user-attachments/assets/aa9c36cc-b0cc-45b9-88f4-36540793bd96

## The math

The first prototype of this MLP was the DoublePerceptron which is just a simplified version with hardcoded layer values.
It has one hidden layer with hardcoded size making it easier to wrap your head around when deriving the maths, i always find that working with
concrete hardcoded values is easier that just going into the abstract example right away.

```py
self.W1 = np.random.randn(16, 784) * np.sqrt(2 / n_features)
self.W2 = np.random.randn(10, 16) * np.sqrt(2 / 16)
```

In this model, the weights are already transposed during initialization, so there is no no need to do that in the forward propagation

```py
self.Z1 = np.dot(self.W1, data) + self.B1  # Z1: (16, m)
self.A1 = self._relu(self.Z1)
self.Z2 = np.dot(self.W2, self.A1) + self.B2  # W2: (16, 10), Z2: (10, m)
self.A2 = self._softmax(self.Z2)
```

Here we can see the exact dimensions we are working with when using the mnist dataset, m is the number of samples. Relu was used as an activation function but later it was switched to tanh. Softmax is used on the output to ensure it is a probability distribution for each digit.

```py
dZ2 = self.A2 - y_true  # (10, m)
dW2 = np.dot(dZ2, self.A1.T) / m  # (10, 16)
dB2 = np.sum(dZ2, axis=1, keepdims=True) / m  # (10, 1)

dZ1 = np.dot(self.W2.T, dZ2) * self._relu_d(self.Z1)  # (16, m)
dW1 = np.dot(dZ1, data.T) / m  # (16, 784)
dB1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (16, 1)
```

Backpropagation is simple as well, it consists of finding how each output value (before and after activation) affects the final loss and moving with the negative gradient of that effect, which will ensure the loss is minimized with respect to weights and biases.

The derivative of the last unactivated layer with respect to loss is just the derivative of the loss function, (mean squared error in this example), evaluated at the output. Then to get Z1 we propagate this error backwards using the weights (Transposed this time), and we have to multiply by the derivative of the activation function because the output is using the activated Z1. I will add more of the exact maths
