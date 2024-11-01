from dataclasses import dataclass
from typing import List
import numpy as np
import solution_utils
import random


@dataclass
class Member:
    path: List[int]
    eval: float
    fitness: float

    def __lt__(self, other: "Member"):
        return self.eval < other.eval

    def __gt__(self, other: "Member"):
        return self.eval > other.eval


def apply_mutations(population: List[Member], mut_chance: float):
    for memb in population:
        if np.random.random() < mut_chance:
            mutate(memb)


def mutate(member: Member):
    path_len = len(member.path)
    idx1, idx2 = np.random.choice(range(1, path_len - 1), 2, replace=False)
    member.path[idx1], member.path[idx2] = member.path[idx2], member.path[idx1]
    member.eval = 0  # set to 0 to recalculate later


# def mutate(member: Member):
#     # randomly switch places in members path
#     path_len = len(member.path)
#     idx1, idx2 = sorted(np.random.choice(path_len - 2, 2, replace=False))
#     idx1, idx2 = idx1 + 1, idx2 + 1
#     subset = member.path[idx1:idx2]
#     np.random.shuffle(subset)
#     member.path[idx1:idx2] = subset


def apply_crossovers(population: List[Member], cross_chance):
    for i in range(0, len(population) - 1, 2):
        if np.random.random() < cross_chance:
            population[i] = crossover(population[i], population[i + 1])


def crossover(member1: Member, member2: Member):
    size = len(member1.path)
    start, end = sorted(np.random.choice(size - 2, 2, replace=False))

    new_path = [-1] * (size - 2)

    # the middle is taken from member1
    new_path[start:end] = member1.path[start + 1 : end + 1]
    current_pos = end
    for city in member2.path[1:-1]:
        if city not in new_path:
            new_path[current_pos] = city
            current_pos = (current_pos + 1) % (size - 2)

    new_path.append(member1.path[-1])
    new_path.insert(0, 0)
    member = Member(new_path, eval=0, fitness=0)
    return member


def create_population(cities_matrix, pop_size) -> List[Member]:
    population = []
    for _ in range(pop_size):
        path = solution_utils.generate_solution(cities_matrix)
        eval = solution_utils.evaluate_solution(cities_matrix, path)
        population.append(Member(path, eval, 0))
    # fill_fitness(population)
    return population


def roulette_selection(population: List[Member]) -> List[Member]:
    total_fitness = sum(memb.fitness for memb in population)
    probs = np.array([memb.fitness / total_fitness for memb in population])
    indices = np.random.choice(
        len(population), size=len(population), p=probs, replace=True
    )
    return [population[ind] for ind in indices]


def tournament_selection(
    population: List[Member], tournament_size: int
) -> List[Member]:
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda member: member.eval)
        selected.append(winner)
    return selected


def fill_fitness(population: List[Member]):
    min_eval = min([mem.eval for mem in population])
    for memb in population:
        # updating scaled fitness for each member
        memb.fitness = 1 - ((memb.eval - min_eval) / min_eval)


class GeneticSolver:

    def __init__(
        self,
        cities_matrix,
        mut_prob=0.1,
        cross_prob=0.75,
        pop_size=2000,
        tournament_size=3,
    ) -> None:
        self.cities_matrix = cities_matrix

        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.pop_size = pop_size
        self.tournament_size = tournament_size

        self.population = create_population(cities_matrix, pop_size)

    def run(self, max_iter=1000) -> List[int]:
        for _ in range(max_iter):
            selected = tournament_selection(self.population, self.tournament_size)
            apply_crossovers(selected, self.cross_prob)
            apply_mutations(selected, self.mut_prob)
            for memb in selected:
                if memb.eval == 0:
                    memb.eval = solution_utils.evaluate_solution(
                        self.cities_matrix, memb.path
                    )

            # fill_fitness(selected)
            self.population = selected

        self.population = sorted(self.population)
        best = self.population[0]
        for memb in reversed(self.population):
            print(memb.path, memb.eval)
        return best.path
