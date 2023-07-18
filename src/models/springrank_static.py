import numpy as np
import scipy.sparse
from os import sys


def springrank_static(
        A: np.ndarray, 
        alpha: float=0.5, 
        l0: float=0.0, 
        l1: float=1.0, 
        verbose: bool=False
) -> np.ndarray:
    """
    Main routine to calculate SpringRank by solving linear system.
    Default parameters are initialized as in the standard SpringRank model as presented in 

    'A physical model for efficient ranking in networks'
    De Bacco C, Larremore D. B., Moore C.

    Parameters:
    -----------
        A:network adjacency matrix (can be weighted)
            type: numpy.array
        alpha: controls the impact of the regularization term
            type: float
        l0: regularization spring's rest length
            type: float
        l1: interaction springs' rest length
            type: float
        verbose: prints additional detail during model execution
            type: bool

    Returns:
    --------
        rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A
            type: numpy.array
    """

    # if not  sparse convert
    A = np.sum(A, axis=0)
    N = A.shape[0]
    k_in = np.sum(A, 0, keepdims=True)
    k_out = np.sum(A, 1, keepdims=True)
    One = np.ones(N)

    C = A + A.T
    D1 = np.zeros(A.shape)
    D2 = np.zeros(A.shape)

    for i in range(A.shape[0]):
        D1[i, i] = k_out[i, 0] + k_in[0, i]
        D2[i, i] = l1 * (k_out[i, 0] - k_in[0, i])

    if alpha != 0.0:
        if verbose:
            print("Using alpha!=0: matrix is invertible")

        B = One * alpha * l0 + np.dot(D2, One)
        A = alpha * np.eye(N) + D1 - C
        L_final = A
        A = scipy.sparse.csr_matrix(np.matrix(A))

        try:
            if verbose:
                print("Switched to scipy.sparse.linalg.bicgstab(A,B)[0]")
            rank = scipy.sparse.linalg.bicgstab(A, B)[0]
            return np.transpose(rank)
        except:
            if verbose:
                print("Trying scipy.sparse.linalg.spsolve(A,B)")
            rank = scipy.sparse.linalg.spsolve(A, B)
            return np.transpose(rank)

    else:
        if verbose:
            print("Using faster computation: fixing a rank degree of freedom")

        C = (
            C
            + np.repeat(A[N - 1, :][None], N, axis=0)
            + np.repeat(A[:, N - 1].T[None], N, axis=0)
        )
        D3 = np.zeros(A.shape)
        for i in range(A.shape[0]):
            D3[i, i] = l1 * (k_out[N - 1, 0] - k_in[0, N - 1])

        B = np.dot(D2, One) + np.dot(D3, One)
        A = scipy.sparse.csr_matrix(np.matrix(D1 - C))
        try:
            if verbose:
                print("Trying scipy.sparse.linalg.bicgstab(A,B)[0]")
            rank = scipy.sparse.linalg.bicgstab(A, B)[0]
            return np.transpose(rank)
        except:
            scipy.sparse.linalg.cond(A) < 1 / sys.float_info.epsilon
            if verbose:
                print("Switched to scipy.sparse.linalg.spsolve")
            rank = scipy.sparse.linalg.spsolve(A, B)
            return np.transpose(rank)
