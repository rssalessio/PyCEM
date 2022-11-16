import numpy as np


def is_positive_definite(x: np.ndarray, atol: float = 1e-9):
    return np.all(np.linalg.eigvals(x) > atol)

def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)