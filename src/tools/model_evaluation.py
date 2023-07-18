import numpy as np
from tools.metrics import sigma_a_metric, sigma_L_metric, beta_a_optimize, beta_L_optimize, agony


def test_dyn_ext(A, s_matrix, beta_a_opt=None, beta_L_opt=None, validation_size=0, test_size=1, step=1):
    results = {}
    results['sigma_a'] = []
    results['sigma_a2'] = []
    results['sigma_L'] = []
    results['accuracy'] = []
    results['agony'] = []
    results['agony2'] = []

    if validation_size != 0:
        beta_a_opt_list = []
        beta_L_opt_list = []
        for v in range(validation_size):
            beta_a_opt_list.append(beta_a_optimize(A[v:v + 1, :, :], s_matrix[v, :]))
            beta_L_opt_list.append(beta_L_optimize(A[v:v + 1, :, :], s_matrix[v, :]))
        beta_a_opt = np.mean(beta_a_opt_list)
        beta_L_opt = np.mean(beta_L_opt_list)

    for i in range(validation_size, A.shape[0] - test_size, step):
        s_test = s_matrix[i, :]
        test = np.sum(A[i:i + test_size, :, :], 0)
        assert test.shape == (A.shape[1], A.shape[1])

        results['beta_a_opt'] = beta_a_opt
        results['beta_L_opt'] = beta_L_opt
        results['sigma_a'].append(sigma_a_metric(test, s_test, beta_a_opt, d=1))
        results['sigma_a2'].append(sigma_a_metric(test, s_test, beta_a_opt, d=2))
        results['sigma_L'].append(sigma_L_metric(test, s_test, beta_L_opt))
        results['accuracy'].append(1 - agony(test, s_test, d=0))
        results['agony'].append(agony(test, s_test, d=1))
        results['agony2'].append(agony(test, s_test, d=2))

    results_extended=dict(results)
    #print(results['sigma_L'])
    results['#'] = len(results['sigma_a'])
    results['beta_a_opt'] = beta_a_opt
    results['beta_L_opt'] = beta_L_opt
    results['sigma_a'] = np.mean(results['sigma_a'])
    results['sigma_a2'] = np.mean(results['sigma_a2'])
    results['sigma_L'] = np.mean(results['sigma_L'])
    results['accuracy'] = np.mean(results['accuracy'])
    results['agony'] = np.mean(results['agony'])
    results['agony2'] = np.mean(results['agony2'])
    return results,results_extended

