from gradient_descent import GradientDescent
from matplt_utils import MatPlotLibUtils
from functions import FunctionTwo, FunctionOne
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def get_startpoint(x_range: Tuple[float, float], y_range: Tuple[float, float]):
    return (
        round(random.uniform(x_range[0], x_range[1]), 2),
        round(random.uniform(y_range[0], y_range[1]), 2),
    )


if __name__ == "__main__":
    # read config from json file
    with open("config.json") as json_data:
        data = json.load(json_data)
        json_data.close()
    tol = data["tol"]
    learning_rate = data["learning_rate"]
    max_iter = data["max_iter"]

    # Function 1
    domain = (-4 * np.pi, 4 * np.pi)
    startpoint = get_startpoint(domain, (0, 0))

    gd = GradientDescent(FunctionOne.gradient, coords=startpoint[0], domain=[domain])
    minimum, steps = gd.run(learning_rate, max_iter, tol)

    print("Start: ", startpoint[0], end="; ")
    print("Minimum funkcji 1: ", minimum, " W ", steps.__len__(), " krokach")

    # MatPlotLibUtils.graph_2d_func(
    #     FunctionOne.function,
    #     x_range=(-4 * np.pi, 4 * np.pi),
    #     point=minimum[0],
    #     steps=steps,
    #     title=f"tolerance={tol}, learning_rate={learning_rate}, steps={steps.__len__()}, start={startpoint[0]}",
    # )

    # Function 2
    domain = (-2, 2), (-2, 2)
    startpoint = get_startpoint(domain[0], domain[1])

    gd = GradientDescent(
        FunctionTwo.gradient,
        coords=startpoint,
        domain=domain,
    )

    minimum, steps = gd.run(learning_rate, max_iter, tol)

    print("Start: ", startpoint, end="; ")
    print("Minimum funkcji 2: ", minimum, " W ", steps.__len__(), " krokach")

    # MatPlotLibUtils.graph_3d_func(FunctionTwo.function, point=minimum, steps=steps)
    MatPlotLibUtils.graph_3d_height_graph(
        FunctionTwo.function,
        point=minimum,
        steps=steps,
        title=f"tolerance={tol}, learning_rate={learning_rate}, steps={steps.__len__()}, , start={startpoint}",
    )
