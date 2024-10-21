from gradient_descent import GradientDescent
from matplt_utils import MatPlotLibUtils
from functions import FunctionTwo, FunctionOne
import random
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def test_generate_f1():
    learning_rates = [0.01 + 0.5 * i for i in range(7)]
    startpoint = 8.00
    max_iter = 1000
    domain = (-4 * np.pi, 4 * np.pi)
    tol = 0.001

    results = []
    for lr in learning_rates:
        gd = GradientDescent(FunctionOne.gradient, coords=startpoint, domain=[domain])
        end_point, steps = gd.run(lr, max_iter, tol)
        results.append((lr, startpoint, end_point, len(steps)))

    fig, ax = plt.subplots(layout="constrained")
    ax.axis("tight")
    ax.axis("off")

    table_data = [["Learning Rate", "Start Point", "End Point", "Steps"]] + results
    table = ax.table(cellText=table_data, cellLoc="center", loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Set column widths

    col_widths = [0.1, 0.2, 0.2, 0.1]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    plt.subplots_adjust(top=0, bottom=0)
    plt.savefig("lab1/images/f1_learn_rate.png", bbox_inches="tight")


def test_generate_f2():
    learning_rates = [0.01 + 0.5 * i for i in range(7)]
    startpoint = (-0.55, -0.5)
    max_iter = 1000
    domain = ((-2, 2), (-2, 2))
    # domain = (-4 * np.pi, 4 * np.pi)
    tol = 0.00001

    results = []
    for lr in learning_rates:
        gd = GradientDescent(FunctionTwo.gradient, coords=startpoint, domain=domain)
        end_point, steps = gd.run(lr, max_iter, tol)
        results.append((lr, startpoint, end_point, len(steps)))

    fig, ax = plt.subplots(layout="constrained")
    ax.axis("tight")
    ax.axis("off")

    table_data = [["Learning Rate", "Start Point", "End Point", "Steps"]] + results
    table = ax.table(cellText=table_data, cellLoc="center", loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Set column widths

    col_widths = [0.1, 0.2, 0.2, 0.1]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    plt.subplots_adjust(top=0, bottom=0)
    plt.savefig("lab1/images/f2_learn_rate.png", bbox_inches="tight")


def test_generate_f2_tol():
    learning_rate = 0.5
    startpoint = (-0.55, -0.5)
    max_iter = 1000
    domain = ((-2, 2), (-2, 2))
    tols = [0.00001 * (10**x) for x in range(4)]

    results = []
    for tol in tols:
        gd = GradientDescent(FunctionTwo.gradient, coords=startpoint, domain=domain)
        end_point, steps = gd.run(learning_rate, max_iter, tol)
        results.append((tol, startpoint, end_point, len(steps)))

    fig, ax = plt.subplots(layout="constrained")
    ax.axis("tight")
    ax.axis("off")

    table_data = [["Tolerance", "Start Point", "End Point", "Steps"]] + results
    table = ax.table(cellText=table_data, cellLoc="center", loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Set column widths

    col_widths = [0.1, 0.2, 0.2, 0.1]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for key, cell in table.get_celld().items():
            if key[1] == i:
                cell.set_width(width)

    plt.subplots_adjust(top=0, bottom=0)
    plt.savefig("lab1/images/f2_tol.png", bbox_inches="tight")
