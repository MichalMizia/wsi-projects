import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, List
import numpy as np


class MatPlotLibUtils:
    @staticmethod
    def graph_2d_func(
        func: Callable[[float], float],
        x_range: Tuple[float, float] = (-2, 2),
        point: Optional[float] = None,
        steps: Optional[List[np.ndarray]] = None,
        title: Optional[str] = None,
    ) -> None:
        min_x, max_x = x_range
        x = np.arange(min_x, max_x, 0.01)
        y = np.array([func(xi) for xi in x])

        plt.figure()
        plt.plot(x, y, label="Function")

        if point:
            py = func(point)
            plt.scatter(point, py, color="r", label="Point")

        if steps:
            y_steps = np.array([func(x[0]) for x in steps])
            plt.plot(steps, y_steps, color="g", marker="o", label="Steps", markersize=2)

        if title:
            plt.title(title, fontsize=10)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    @staticmethod
    def graph_3d_func(
        func: Callable[[float, float], float],
        x_range: tuple[float, float] = (-2, 2),
        y_range: tuple[float, float] = (-2, 2),
        point: Optional[np.ndarray] = None,
        steps: Optional[List[np.ndarray]] = None,
        title: Optional[str] = None,
    ) -> None:
        min_x, max_x = x_range
        min_y, max_y = y_range
        x = np.arange(min_x, max_x, 0.01)
        y = np.arange(min_y, max_y, 0.01)
        X, Y = np.meshgrid(x, y)

        Z = func(X, Y)  # type: ignore

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if point is not None:
            px, py = point
            pz = func(px, py)
            ax.scatter(px, py, pz, color="r")

        if steps and steps.__len__():
            new_steps = np.array(steps)
            x_steps, y_steps = new_steps[:, 0], new_steps[:, 1]
            z_steps = np.array([func(x, y) for x, y in zip(x_steps, y_steps)])
            ax.plot(x_steps, y_steps, z_steps, color="b", marker="o", markersize=2)

        if title:
            plt.title(title, fontsize=10)

        ax.plot_surface(X, Y, Z)  # type: ignore
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    @staticmethod
    def graph_3d_height_graph(
        func: Callable[[float, float], float],
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        point: Optional[np.ndarray] = None,
        steps: Optional[List[np.ndarray]] = None,
        title: Optional[str] = None,
    ) -> None:
        min_x, max_x = x_range
        min_y, max_y = y_range
        x = np.arange(min_x, max_x, 0.01)
        y = np.arange(min_y, max_y, 0.01)
        X, Y = np.meshgrid(x, y)

        Z = func(X, Y)  # type: ignore

        fig, ax = plt.subplots()
        contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")
        fig.colorbar(contour)

        if point is not None:
            px, py = point
            ax.scatter(px, py, color="g", label="Point")

        if steps is not None:
            new_steps = np.array(steps)
            xs, ys = new_steps[:, 0], new_steps[:, 1]
            ax.plot(xs, ys, color="r", marker="o", label="Steps", markersize=2)

        if title:
            plt.title(title, fontsize=10)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
