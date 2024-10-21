from gradient_descent import GradientDescent
import numpy as np
from matplt_utils import MatPlotLibUtils
import random
from functions import FunctionTwo, FunctionOne
from typing import Tuple


def get_startpoint(x_range: Tuple[float, float], y_range: Tuple[float, float]):
    return (
        random.uniform(x_range[0], x_range[1]),
        random.uniform(y_range[0], y_range[1]),
    )


if __name__ == "__main__":
    # Funkcja 1
    domain = (-4 * np.pi, 4 * np.pi)
    startpoint = get_startpoint(domain, (0, 0))
    print("Start: ", startpoint[0])

    gd = GradientDescent(
        FunctionOne.gradient, coords=startpoint[0], domain=[domain], learning_rate=0.5
    )
    minimum, steps = gd.run()
    print("Minimum funkcji 1: ", minimum, " W ", steps.__len__(), " krokach")

    # MatPlotLibUtils.graph_2d_func(
    #     FunctionOne.function,
    #     x_range=(-4 * np.pi, 4 * np.pi),
    #     point=minimum[0],
    #     steps=steps,
    # )

    # Funkcja 2
    domain = (-2, 2), (-2, 2)
    startpoint = get_startpoint(domain[0], domain[1])
    print("Start: ", startpoint)

    gd = GradientDescent(
        FunctionTwo.gradient,
        coords=startpoint,
        domain=domain,
        learning_rate=3,
        tol=0.01,
    )

    minimum, steps = gd.run()
    print("Minimum funkcji 2: ", minimum, " W ", steps.__len__(), " krokach")

    # MatPlotLibUtils.graph_3d_func(FunctionTwo.function, point=minimum, steps=steps)
    MatPlotLibUtils.graph_3d_height_graph(
        FunctionTwo.function, point=minimum, steps=steps
    )
