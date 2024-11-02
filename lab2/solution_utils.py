import numpy as np
import random


def decode_solution(cities_matrix, solution):
    return list(map(lambda city_id: cities_matrix.index[city_id], solution))


def validate_solution(cities_matrix, solution):
    # check if each city is visited exactly one time
    assert len(list(solution)) == len(set(solution))
    assert sorted(solution) == list(range(len(cities_matrix)))
    # check if start and finish cities are in the correct place
    assert solution[0] == 0 and solution[-1] == len(cities_matrix) - 1


def evaluate_solution(cities_matrix, solution):
    total_distance = 0
    for city_id in range(len(solution) - 1):
        total_distance += cities_matrix.iloc[solution[city_id], solution[city_id + 1]]
    return total_distance


def generate_solution(cities_matrix):
    return (
        [0]
        + np.random.permutation(np.arange(1, len(cities_matrix) - 1)).tolist()
        + [len(cities_matrix) - 1]
    )


def minmax(nums):
    min = nums[0]
    max = nums[0]
    for num in nums:
        if num < min:
            min = num
        elif num > max:
            max = num
    return min, max


def merge_paths(path_a, path_b):
    result = []
    start = random.randint(0, len(path_a) - 1)
    finish = random.randint(start + 1, len(path_a))
    sub_path_from_a = path_a[start:finish]
    remaining_path_from_b = list(
        [item for item in path_b if item not in sub_path_from_a]
    )
    for i in range(0, len(path_a)):
        if start <= i < finish:
            result.append(sub_path_from_a.pop(0))
        else:
            result.append(remaining_path_from_b.pop(0))
    return result


good_sol = [
    0,
    37,
    32,
    29,
    17,
    38,
    10,
    9,
    26,
    23,
    28,
    15,
    8,
    3,
    7,
    44,
    46,
    36,
    30,
    27,
    14,
    18,
    47,
    25,
    19,
    33,
    48,
    5,
    22,
    35,
    6,
    31,
    45,
    24,
    16,
    13,
    21,
    20,
    12,
    49,
    11,
    40,
    4,
    43,
    39,
    2,
    1,
    34,
    41,
    42,
    50,
]

if __name__ == "__main__":
    print(merge_paths([0, 1, 4, 3, 2], [0, 3, 4, 1, 2]))
    pass
