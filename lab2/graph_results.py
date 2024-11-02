import matplotlib.pyplot as plt
import solution_utils
import numpy as np
from ga_tournament import GeneticSolver

# from ga_roulette import GeneticSolver


def graph_results(cities_matrix):
    population_sizes = [i * 50 for i in range(1, 10)]
    results = []
    repetitions = 5

    for pop_size in population_sizes:
        best_evals = []

        for _ in range(repetitions):
            solver = GeneticSolver(
                cities_matrix,
                pop_size=pop_size,
            )
            solver.run()
            best_eval = solver.population[0].eval
            best_evals.append(best_eval)

        best_eval = min(best_evals)
        avg_best_eval = np.mean(best_evals)
        std_dev = np.std(best_evals)
        results.append((pop_size, best_eval, avg_best_eval, std_dev))

    # display result in matplotlib
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    table_data = [
        ["Population Size", "Best Eval", "Avg Best Eval", "Std Dev"]
    ] + results
    table = ax.table(
        cellText=table_data, colLabels=None, cellLoc="center", loc="center"
    )

    plt.title(
        "Effect of Population Size on GA with 100 iterations, mut_prob = 0.01, cross_prob=0.75"
    )
    plt.show()
