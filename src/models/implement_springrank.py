import numpy as np
from typing import Tuple, Union, Optional

from models.springrank_static import springrank_static
from tools.metrics import (
    beta_a_optimize,
    beta_L_optimize,
    sigma_a_metric,
    sigma_L_metric,
    agony,
)


def static_sr(
    A: np.ndarray, 
    validation_size: int=20, 
    test_size: int=1, 
    end_training: Optional[int]=None, 
    sigma_d: int=1, 
    agony_d: int=1
) -> Tuple[dict, dict]:
    """
    Implementation of SpringRank model given an adjaceny matrix with a time component

    Parameters:
    -----------
    A: adjacency matrix with a dimension of (T, N, N) where T is the number of time steps and N is the number of nodes
        type: numpy.array
    validation_size: size of the validation set
        type: int
    test_size: size of the test set
        type: int
    end_training: time step at which training ends
        type: int
    sigma_d: argument to sigma_a metric
        type: int
    agony_d: argument to agony metric
        type: int

    Returns:
    --------
    list of performance results
    """
    s0 = springrank_static(A[: validation_size // 2, :, :])
    beta_a = beta_a_optimize(A[validation_size // 2 : validation_size, :, :], s0)
    beta_L = beta_L_optimize(A[validation_size // 2 : validation_size, :, :], s0)

    T = A.shape[0]
    N = A.shape[1]

    s_matrix = np.zeros((T - end_training - test_size, N))

    sigma_a_list = []
    sigma_L_list = []
    accuracy_list = []
    agony_list = []

    for i in range(end_training, T - test_size, 1):
        test = np.sum(A[i : i + test_size, :, :], 0)
        s_matrix[i - end_training, :] = springrank_static(A[:i, :, :])

        sigma_a_list.append(
            sigma_a_metric(test, s_matrix[i - end_training, :], beta_a, d=sigma_d)
        )
        sigma_L_list.append(sigma_L_metric(test, s_matrix[i - end_training, :], beta_L))
        accuracy_list.append(1.0 - agony(test, s_matrix[i - end_training, :], d=0))
        agony_list.append(agony(test, s_matrix[i - end_training, :], d=agony_d))

    results = {
        "sigma_a": np.mean(sigma_a_list),
        "sigma_L": np.mean(sigma_L_list),
        "accuracy": np.mean(accuracy_list),
        "agony": np.mean(agony_list),
        "#": len(accuracy_list),
    }

    results_extended = {
        "sigma_a": sigma_a_list,
        "sigma_L": sigma_L_list,
        "accuracy": accuracy_list,
        "agony": agony_list,
        "#": len(accuracy_list),
    }

    return results, results_extended

