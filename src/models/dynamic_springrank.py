import numpy as np
import scipy
import math
import scipy.sparse.linalg

from typing import Any,Tuple,Union,Optional
from tools.model_evaluation import test_dyn_ext
from scipy.linalg import block_diag
from models.springrank_static import springrank_static
from tools.create_intervals import create_interval_positive, create_interval


def dynamic_springrank(
    A: np.ndarray,
    end_training: Optional[int]=None,
    validation_size: int=4,
    test_size: int=1,
    initialization_window: int=5,
    start: int=10,
    s0: Optional[np.ndarray]=None,
    k0: Optional[float]=None,
    beta_L_opt: Optional[float]=None,
    beta_a_opt: Optional[float]=None,
    verbose: bool=False,
) -> Union[Tuple[dict, Any, dict], Tuple[dict, dict, np.ndarray]]:
    """
        Implementaion of Dynamic SpringRank as presented in

        "A model for efficient dynamical ranking in networks"
        Della Vecchia A., Neocosmos K., Larremore D. B., Moore C., De Bacco C.
        
        (referred to as Online Dynamic SpringRank in paper)
        Note: Runs on either train or test set

        Parameters:
        -----------
        A: adjacency matrix with dimension (T, N, N) where T is the number of time steps and N is the number of nodes
            type: numpy.array
        end_training: time step at which training ends
            type: int
        validation_size: size of validation set
            type: int
        test_size: size of the test set
            type: int
        initialization_window: number of time steps used to deterime s0 during training
            type: int
        start: time step at which to start evaluating scores
            type: int
        s0: initial score for nodes
            type: numpy.array
        k0: hyperparameter in model (refer to model equation in paper)
            type: float
        beta_L_opt: optimal value of beta for sigma_L metric
            type: float
        beta_a_opt: optimal value of beta for sigma_a metric
            type: float
        verbose: prints additional detail of executed parts of model
            type: bool

        Returns:
        --------
        results_train: training results based on optimal k0
            type: dict
        dict_k0: training results based on different k0
            type: dict
        results: test results
            type: dict
        results_extended: test results for each test set interval (determined by test_size
            type: dict
        s_matrix_test: scores matrix of dimension (T, N) containing score of each node at each time step
            type: numpy.array
    """
    if (s0 is None) or (k0 is None):  # then skip training and go straight to test
        ############################# Training ##################################

        if verbose:
            print("Training the model")
        A_init = A[:initialization_window, :, :]
        A_train = A[initialization_window:end_training, :, :]
        T = A_train.shape[0]
        N = A_train.shape[1]
        dict_k0 = {}
        s0 = springrank_static(A_init, l0=-0.0)
        K = {0: [0.01, 0.1, 1, 10, 100, 1000]}
        count_total = 3  # determines number of intervals used in grid-search

        assert len(s0) == N
        count = 0
        while count < count_total:
            k0_list = K[count]
            for k0 in k0_list:
                if verbose:
                    print("k0:", k0)

                s_matrix = np.zeros((T + 1, N))
                s_matrix[0, :] = s0
                for t in range(T):
                    s_matrix[t + 1, :] = springrank_static(
                        A_train[t : t + 1, :, :], alpha=N * k0, l0=s_matrix[t, :]
                    )

                dict_k0["k0_{}".format(k0)], _ = test_dyn_ext(
                    A_train[start:, :, :],
                    s_matrix[start:, :],
                    validation_size=validation_size,
                    test_size=test_size,
                )
                dict_k0["k0_{}".format(k0)]["k0"] = k0
                dict_k0["k0_{}".format(k0)]["s_matrix"] = s_matrix

            sorted_res_k0 = sorted(
                dict_k0.keys(),
                key=lambda k: np.mean(dict_k0[k]["accuracy"]),
                reverse=True,
            )
            optimal_k0 = float(sorted_res_k0[0][3:])
            if verbose:
                print("Optimal k0:", optimal_k0)
            interval = create_interval_positive(optimal_k0, -count)
            K[count + 1] = interval
            count += 1

        sorted_res_k0 = sorted(
            dict_k0.keys(), key=lambda k: np.mean(dict_k0[k]["accuracy"]), reverse=True
        )
        results_train = dict_k0[sorted_res_k0[0]]

        return results_train, _, dict_k0

    else:
        ############################# Testing ##################################

        T, N = A.shape[0], A.shape[1]
        beta_a_opt = beta_a_opt
        beta_L_opt = beta_L_opt
        k0 = k0

        s_matrix_test = np.zeros((T + 1, N))
        s_matrix_test[0, :] = s0
        for t in range(T):
            s_matrix_test[t + 1, :] = springrank_static(
                A[t : t + 1, :, :], alpha=N * k0, l0=s_matrix_test[t, :]
            )
        results, results_extended = test_dyn_ext(
            A,
            s_matrix_test,
            beta_a_opt=beta_a_opt,
            beta_L_opt=beta_L_opt,
            test_size=test_size,
        )
        results["k0"] = k0

        return results, results_extended, s_matrix_test

def dsr_offline(
    A: np.ndarray,
    k0: Optional[float]=None,
    l1: float=1.0,
    end_training: Optional[int]=None,
    validation_size: int=4,
    test_size: int=1,
    start: int=10,
    verbose: bool=False,
) -> Tuple[dict, dict, np.ndarray, dict, dict]:
    """
    Alternative implementation of Dynamic SpringRank model as presented in 
    
    "A model for efficient dynamical ranking in networks"
    Della Vecchia A., Neocosmos K., Larremore D. B., Moore C., De Bacco C.
        
    (Referred to as Offline Dynamics SpringRank in paper)
    NB: performance evaluation for training is the same as testing

    Parameters:
    ----------
        A: adjacency matrix with dimension (T,N,N) where T is the number of time-steps and N is the number of nodes
            type: numpy.ndarray
        k0: float which influences the effect of the self-springs
            type: float
        l1: rest-length of interaction springs
            type: float
        end_training: number of time-steps involved in training
            type: int
        validation_size: size of the validation set
            type: int
        test_size: size of the test set
            type: int
        start: time step at which to start evaluating scores
            type: int
        verbose: includes more detail of part of the model being executed
            type: bool
    Returns:
    -------
        results: test results
            type: dict
        results_extended: test results for each test set interval (determined by test_size
            type: dict
        score_test: scores matrix of dimension (T_test, N) containing score of each node at each time step of test set
            type: numpy.array
        results_train: training results based on optimal k0
            type: dict
        dict_k0: training results based on different k0
            type: dict 
    """

    end_training = end_training  # int(A.shape[0]*(train_percent/100))
    A_train = A[:end_training, :, :]
    A_test = A[end_training:, :, :]

    ########################## Training #############################

    if k0 is None:
        K = {0: [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        count_total = 3  # determines number of intervals used in grid-search

    else:
        K = {0: [k0]}
        count_total = 1

    dict_k0 = {}
    # CONSTRUCT INPUTS FOR SOLVER
    N = A_train.shape[1]  # number of nodes
    T = A_train.shape[0]  # number of time-steps
    l1 = np.full((N, 1), l1)
    D_out = np.zeros((T, N, N))  # weighted out-degree matrix
    D_in = np.zeros((T, N, N))  # weight in-degree matrix
    for t in range(T):
        for i in range(N):
            D_in[t, i, i] = np.sum(A_train[t, :, i])
            D_out[t, i, i] = np.sum(A_train[t, i, :])

    b = np.dot((D_out - D_in), l1)
    B = b.flatten()  # create a vector from b

    if verbose and k0 is None:
        print("Finding suitable k0")

    count = 0
    while count < count_total:
        k0_list = K[count]
        for k0 in k0_list:
            if verbose:
                print("k0:", k0)
            l = (
                2 * N * k0 * np.broadcast_to(np.eye(N), (T, N, N))
                + D_out
                + D_in
                - (A_train + A_train.transpose((0, 2, 1)))
            )
            L = block_diag(*l)  # create a block diagonal matrix

            # create block matrix the same size as L with identity matrix*N*k0 under the diagonal block
            under_diag = np.concatenate(
                (-N * k0 * np.eye(N * T - N), np.zeros((N * T - N, N))), axis=1
            )
            under_diag = np.concatenate((np.zeros((N, N * T)), under_diag), axis=0)

            # create block matrix the same size as L with identity matrix*N*k0 over the diagonal block
            over_diag = np.concatenate(
                (np.zeros((N * T - N, N)), -N * k0 * np.eye(N * T - N)), axis=1
            )
            over_diag = np.concatenate((over_diag, np.zeros((N, N * T))), axis=0)

            L_final = L + under_diag + over_diag
            score_train = np.zeros((T + 1, N))
            score_train[0, :] = np.zeros((1, N))  # initial score
            for t in range(1, T):
                L_csr = scipy.sparse.csr_matrix(
                    L_final[: N * t, : N * t]
                )  # convert L_final to a compressed sparse row matrix
                B_final = B[: N * t]

                #### SOLVER ####
                try:
                    if verbose:
                        print("Using bicgstab")
                    score = scipy.sparse.linalg.bicgstab(L_csr, B_final)[0]
                    if verbose:
                        print("Finished bicgstab")
                except:
                    if verbose:
                        print("Trying to use spsolve")
                    score = scipy.sparse.linalg.spsolve(L_csr, B_final)
                    if verbose:
                        print("Successfully used spsolve")
                assert not np.isnan(
                    score[-N:].sum()
                ), f"One or more score is NaN for k0={k0}"  # Check if NaN is in array
                score_train[t + 1, :] = score[-N:]
            #### RESULTS ####
            # score_train = score_train.reshape((T, N)) #reshaping to work with test_dyn_ext function
            if verbose:
                print("Evaluating training set starts!")
            dict_k0["k0_{}".format(k0)], _ = test_dyn_ext(
                A_train[start:, :, :],
                score_train[start:, :],
                validation_size=validation_size,
                test_size=test_size,
            )
            dict_k0["k0_{}".format(k0)]["k0"] = k0
            dict_k0["k0_{}".format(k0)]["score_train"] = score_train
            if verbose:
                print("Evaluating training set ends!")
            # print(dict_k0)

        sorted_res_k0 = sorted(
            dict_k0.keys(), key=lambda k: np.mean(dict_k0[k]["accuracy"]), reverse=True
        )  # sorting results according to accuracy, largest value first
        optimal_k0 = float(sorted_res_k0[0][3:])
        interval = create_interval_positive(optimal_k0, -count)
        K[count + 1] = interval
        count += 1
    sorted_res_k0 = sorted(
        dict_k0.keys(), key=lambda k: np.mean(dict_k0[k]["accuracy"]), reverse=True
    )  # sorting results according to accuracy, largest value first
    # print(sorted_res_k0)
    results_train = dict_k0[sorted_res_k0[0]]

    # Inputs of testing
    k0 = results_train["k0"]
    beta_a_opt = results_train["beta_a_opt"]
    beta_L_opt = results_train["beta_L_opt"]

    ########################### Testing ################################

    # CONSTRUCT INPUTS FOR SOLVER
    N = A_test.shape[1]  # number of nodes
    T_test = A_test.shape[0]  # number of time-steps
    l1 = np.full((N, 1), l1)
    D_out = np.zeros((T_test, N, N))  # weighted out-degree matrix
    D_in = np.zeros((T_test, N, N))  # weight in-degree matrix
    for t in range(T_test):
        for i in range(N):
            D_in[t, i, i] = np.sum(A_test[t, :, i])
            D_out[t, i, i] = np.sum(A_test[t, i, :])

    b_test = np.dot((D_out - D_in), l1)
    B_test = b_test.flatten()  # create a vector from b

    l = (
        2 * N * k0 * np.broadcast_to(np.eye(N), (T_test, N, N))
        + D_out
        + D_in
        - (A_test + A_test.transpose((0, 2, 1)))
    )

    score_test = np.zeros((T_test + 1, N))
    score_test[0, :] = results_train["score_train"][-1, :]
    for t in range(A_test.shape[0]):
        # Progressively expand block matrix to include more time steps
        cols = L_final.shape[1]
        rows = L_final.shape[0]
        L_biggest = np.zeros((rows + N, cols + N))
        L_biggest[:rows, :cols] = L_final
        L_biggest[-2 * N : -N, cols:] = -N * k0 * np.eye(N)
        L_biggest[rows:, -2 * N : -N] = -N * k0 * np.eye(N)
        L_biggest[rows:, -N:] = l[t, :, :]
        L_final = L_biggest
        L_csr = scipy.sparse.csr_matrix(
            L_final
        )  # convert L_final to a compressed sparse row matrix

        B_final = np.concatenate((B, B_test[: N * (t + 1)]), axis=0)

        if verbose:
            print("Finding scores_test:")
        #### SOLVER #####
        try:
            if verbose:
                print("Using bicgstab")
            score = scipy.sparse.linalg.bicgstab(L_csr, B_final)[0]
            if verbose:
                print("Finished bicgstab", "\n")
        except:
            if verbose:
                print("Trying to use spsolve")
            score = scipy.sparse.linalg.spsolve(L_csr, B_final)
            assert math.isnan(score[0]) == False
            if verbose:
                print("Successfully used spsolve", "\n")
        assert not np.isnan(
            score[-N:].sum()
        ), "One or more score is NaN"  # Check if NaN is in array
        score_test[t + 1, :] = score[-N:]

    #### RESULTS ####
    score_test = score_test.reshape(
        (T_test + 1, N)
    )  # reshaping to work with test_dyn_ext function
    if verbose:
        print("Evaluating test set starts!")
    results, results_extended = test_dyn_ext(
        A_test,
        score_test,
        beta_a_opt=beta_a_opt,
        beta_L_opt=beta_L_opt,
        test_size=test_size,
    )
    results["k0"] = k0
    if verbose:
        print("Evalutating test set ended!")

    return results, results_extended, score_test, results_train, dict_k0
