from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Callable, List, Tuple
from math import ceil
from utils import is_positive_definite, is_symmetric

class Population(ABC):
    def __init__(self, dim: int):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._size

    @abstractmethod
    def sample(self, num_points: int) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> Population:
        raise NotImplementedError

    @abstractmethod
    def update(self, elite_points: List[Tuple[float, np.ndarray]]):
        raise NotImplementedError


class GaussianPopulation(Population):
    means: np.ndarray
    covariance: np.ndarray

    def __init__(self, means: np.ndarray, covariance: np.ndarray):
        super().__init__(self.means.shape[0])
        assert len(means.shape) <= 1, 'Means should be a scalar or a vector'
        assert means.shape[0] == covariance.shape[0], 'Means and std_devs should have the same dimensionality'
        assert is_positive_definite(covariance) and is_symmetric(covariance), 'Covariance needs to be PSD and symmetric'

        self.means = means
        self.covariance = covariance

    def clone(self) -> GaussianPopulation:
        return GaussianPopulation(self.means.copy(), self.std_devs.copy(), self.min_std_dev)

    def sample(self, num_points: int) -> List[np.ndarray]:
        return np.random.multivariate_normal(self.means, cov=self.covariance, size=(num_points,))

def evaluate_population(self,
    fun: Callable[[np.ndarray], float],
    population: Population,
    num_points: int,
    elite_fraction: float = 0.2):
    assert elite_fraction > 0 and elite_fraction < 1, 'Elite fraction needs to be in (0,1)'

    # Sample population
    points = population.sample(num_points)

    # Evaluate points
    results = map(fun, points)

    # Sort results
    sorted_results = sorted(zip(results, points), key=lambda x: x[0])

    # Compute elite population
    elite_results = sorted_results[-ceil(elite_fraction * num_points):]

    # Update population
    new_population = population.clone()
    new_population.update(elite_results)

    return new_population

