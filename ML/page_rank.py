import numpy as np
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt
from typing import Callable
from multiprocessing import Process

def random_adjacency_matrix(n: int, p: float = .5) -> np.ndarray:
    A = (np.random.rand(n * n) > p).reshape((n, n))
    np.ravel(A)[::n+1] = False  # Remove self loops

    for j in it.compress(range(A.shape[1]), range(A.shape[1])):
        if j != 0:
            A[j - 1, j] = True
        else:
            A[j + 1, j] = True

    return A


def normalize(A: np.ndarray) -> np.ndarray:
    return A / A.sum(axis=0)


# Examples
# visualize_matrix(10, .5, lambda: plt.pause(1), cmap='bwr') <- change cmap and pause for one second
# visualize_matrix(10, .3, show, cmap='binary', linewidth=.1) <- press enter for next
# visualize_matrix(10, .3, show, cmap='binary', linewidth=.1, annot=True) <- Showing numbers


def visualize_matrix(n: int, p: float, sleep_function: Callable[[None], None], **kwargs) -> np.ndarray:
    """ 
        n: number of vertices in page rank graph,
        p: probability of an edge between any two vertices in the graph (except for self)
        sleep_function: function to call between iterations. For example, input would pause until a button was clicked
        or lambda: plt.pause(1) to pause for a second

        kwargs args forwarded to sns.heatmap, annot=True will put numbers on cells, linewidth = nonzero will draw grid 
        lines

        returns number of iterations taken to converge
    """

    plt.clf()

    A = normalize(random_adjacency_matrix(n, 1.0 - p))

    B = A
    for i in range(100):
        prev_b = B.copy()

        sns.heatmap(B, **kwargs)
        plt.xticks([])
        plt.yticks([])

        B = A @ B
        sleep_function()

        if np.all(np.isclose(prev_b, B)):
            break

        plt.clf()

    return A, i
