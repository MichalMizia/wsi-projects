import argparse
import pathlib
import numpy as np
import pandas as pd
from solution_utils import (
    decode_solution,
    good_sol,
)
from visualizer import visualize
from graph_results import graph_results

from ga_roulette import GeneticSolver

# from ga_tournament import GeneticSolver


MINI_CITIES_NUM = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cities-path",
        type=pathlib.Path,
        required=True,
        help="Path to cities csv file",
    )
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="full",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Warszawa")
    parser.add_argument("--finish", type=str, default="Rzeszów")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[
        ~((data.index == args.finish) | (data.index == args.start))
    ]
    if args.problem_size == "mini":
        city_names = (
            [args.start]
            + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist()
            + [args.finish]
        )
    else:
        city_names = (
            [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]
        )

    return data[city_names].loc[city_names]


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)

    # graph_results(data)
    # visualize(decode_solution(data, good_sol))
    solver = GeneticSolver(data, pop_size=200)
    sol = solver.run(max_iter=200)

    print(sol, decode_solution(data, sol))

    if args.visualize:
        visualize(decode_solution(data, sol))


if __name__ == "__main__":
    main()
