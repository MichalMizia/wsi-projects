import numpy as np
from typing import Tuple


# numer indeksu: 331407, A = 7, B = 4, C = 1
class FunctionOne:
    @staticmethod
    def function(x: float):
        return 7 * x + 4 * np.sin(x)

    @staticmethod
    def x_derivative(x: float) -> float:
        # partial derivative with respect to x
        return 7 + 4 * np.cos(x)

    @staticmethod
    def gradient(x: float) -> float:
        return FunctionOne.x_derivative(x)


class FunctionTwo:
    @staticmethod
    def function(x: float, y: float):
        return (x * y) / np.exp(x**2 + y**2)

    @staticmethod
    def x_derivative(x: float, y: float) -> float:
        # partial derivative with respect to x
        return (y * (1 - 2 * x**2)) / np.exp(x**2 + y**2)

    @staticmethod
    def y_derivative(x: float, y: float) -> float:
        # partial derivative with respect to y
        return (x * (1 - 2 * y**2)) / np.exp(x**2 + y**2)

    @staticmethod
    def gradient(coords: Tuple[float, float]) -> Tuple[float, float]:
        x, y = coords
        return FunctionTwo.x_derivative(x, y), FunctionTwo.y_derivative(x, y)
