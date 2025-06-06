from typing import Callable, Tuple, List, Union
import numpy as np


class GradientDescent:
    def __init__(
        self,
        gradient: Callable,
        coords,
        domain,
    ):
        if isinstance(coords, float):
            self.coords = np.array([coords])
        else:
            self.coords = np.array(coords)

        self.gradient = gradient
        self.domain = np.array(domain)

    @staticmethod
    def get_clamped_coords(coords, domain):
        clamped_coords = np.copy(coords)
        for i in range(len(domain)):
            clamped_coords[i] = np.clip(coords[i], domain[i][0], domain[i][1])
        return clamped_coords

    def run(
        self,
        learning_rate: float = 0.5,
        max_iter: int = 1000,
        tol: float = 0.001,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        coords = self.coords
        steps: List[np.ndarray] = [coords]

        for _ in range(max_iter):
            gradient_val = np.array(self.gradient(coords))
            if np.linalg.norm(gradient_val) < tol:
                break

            new_coords = coords - gradient_val * learning_rate
            new_coords = self.get_clamped_coords(new_coords, self.domain)

            # Jeśli współrzędne sie nie zmieniły to wychodzimy z algorytmu
            if np.array_equal(new_coords, coords):
                break
            coords = new_coords

            steps.append(coords)

        return coords, steps
