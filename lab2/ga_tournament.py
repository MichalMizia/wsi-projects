from dataclasses import dataclass
from typing import List
import numpy as np
import solution_utils
import random


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


def apply_crossovers(population: List[Member], cross_chance):
    # apply crossover to each 'pair' in the population
    for i in range(0, len(population) - 1, 2):
        if np.random.random() < cross_chance:
            population[i], population[i + 1] = crossover(
                population[i], population[i + 1]
            ), crossover(population[i + 1], population[i])


def crossover(member1: Member, member2: Member):
    size = len(member1.path)
    start, end = sorted(np.random.choice(size - 2, 2, replace=False))

    new_path = [-1] * (size - 2)  # without start and end

    # the middle is taken from member1
    new_path[start:end] = member1.path[start + 1 : end + 1]
    current_pos = end
    for city in member2.path[1:-1]:
        if city not in new_path:
            new_path[current_pos] = city
            current_pos = (current_pos + 1) % (size - 2)

    new_path.append(member1.path[-1])  # insert start and end point
    new_path.insert(0, 0)
    member = Member(
        new_path,
        eval=0,
    )
    return member


def create_population(cities_matrix, pop_size) -> List[Member]:
    population = []
    for _ in range(pop_size):
        path = solution_utils.generate_solution(cities_matrix)
        eval = solution_utils.evaluate_solution(cities_matrix, path)
        population.append(Member(path, eval))
    return population


def tournament_selection(
    population: List[Member], tournament_size: int
) -> List[Member]:
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda member: member.eval)
        selected.append(winner)
    return selected


class GeneticSolver:

    def __init__(
        self,
        cities_matrix,
        pop_size=200,
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
        max_iter=100,
        mut_prob=0.1,
        cross_prob=0.75,
        tournament_size=5,
    ) -> List[int]:
        for _ in range(max_iter):
            selected = tournament_selection(self.population, tournament_size)
            apply_crossovers(selected, cross_prob)
            apply_mutations(selected, mut_prob)
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
