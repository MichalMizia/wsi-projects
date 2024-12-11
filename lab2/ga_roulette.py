from dataclasses import dataclass
from typing import List
import numpy as np
import solution_utils


@dataclass
class Member:
    path: List[int]
    eval: float

    def __lt__(self, other: "Member"):
        return self.eval < other.eval

    def __gt__(self, other: "Member"):
        return self.eval > other.eval


def apply_mutations(population: List[Member], mut_chance: float):
    path_len = len(population[0].path)
    # for each member, there is a small chance that 2 cities in the path will switch
    for member in population:
        if np.random.random() < mut_chance:
            idx1, idx2 = np.random.choice(range(1, path_len - 1), 2, replace=False)
            member.path[idx1], member.path[idx2] = member.path[idx2], member.path[idx1]
            member.eval = 0

    return population


def apply_crossovers(population: List[Member], cross_chance: float) -> List[Member]:
    new_population = []
    for i in range(0, len(population)):
        parent1 = population[i]
        parent2 = population[np.random.choice(len(population))]
        if np.random.random() < cross_chance:
            # Perform crossover and get two offspring
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([offspring1, offspring2])
        else:
            # add copies of parents to new population
            new_population.extend([parent1, parent2])

    return new_population


def crossover(parent1: Member, parent2: Member):
    size = len(parent1.path)

    child1 = [-1] * size
    child2 = [-1] * size

    start = np.random.randint(0, size - 1)
    end = np.random.randint(start + 1, size - 1) if start + 2 != size else size

    child1[start:end] = parent1.path[start:end]
    child2[start:end] = parent2.path[start:end]

    # Fill the remaining positions
    def fill_child(child, parent_other):
        parent_pos = 0
        for current_pos in range(size):
            if child[current_pos] == -1:
                while parent_other.path[parent_pos % size] in child:
                    parent_pos += 1
                child[current_pos] = parent_other.path[parent_pos % size]
                parent_pos += 1

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return Member(child1, 0), Member(child2, 0)


def create_population(cities_matrix, pop_size) -> List[Member]:
    population = []
    for _ in range(pop_size):
        path = solution_utils.generate_solution(cities_matrix)
        eval = solution_utils.evaluate_solution(cities_matrix, path)
        population.append(Member(path, eval))
    return population


def roulette_selection(population: List[Member]) -> List[Member]:
    min_eval = min(memb.eval for memb in population)
    shift = -min_eval + 10  # Shift to not divide by 0``
    shift = -min_eval + 1e+2  # Shift evaluations to be positive
    fitnesses = [1 / (memb.eval + shift) for memb in population]
    total_fitness = np.sum(fitnesses)
    probs = fitnesses / total_fitness

    indices = np.random.choice(
        len(population), size=len(population) // 2, p=probs, replace=True
    )
    return [population[ind] for ind in indices]


class GeneticSolver:

    def __init__(
        self,
        cities_matrix,
        pop_size=100,
        population=None,
    ) -> None:
        self.cities_matrix = cities_matrix
        self.pop_size = pop_size

        if population is not None:
            self.population = population
        else:
            self.population = create_population(cities_matrix, pop_size)

    def run(
        self,
        max_iter=200,
        mut_prob=0.05,
        cross_prob=0.9,
        tournament_size=5,
    ) -> List[int]:
        for _ in range(max_iter):
            selected = roulette_selection(self.population)
            selected = apply_crossovers(selected, cross_prob)
            selected = apply_mutations(selected, mut_prob)

            for memb in selected:
                if memb.eval == 0:
                    memb.eval = solution_utils.evaluate_solution(
                        self.cities_matrix, memb.path
                    )

            self.population = selected

        self.population = sorted(self.population)
        best = self.population[0]
        for memb in reversed(self.population):
            print(memb.path, memb.eval)

        return best.path
