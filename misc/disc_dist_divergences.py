import numpy as np
from scipy.optimize import linear_sum_assignment


def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-6) / (q + 1e-6)))


def symm_kl_divergence(p, q):
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))


def minimum_hamming_distance(z1, z2):

    # check if they got the same shape
    assert z1.shape == z2.shape

    # reshape and get bins
    rz1 = z1.reshape(-1)
    rz2 = z2.reshape(-1)

    # get min length
    length = np.maximum(np.max(rz1), np.max(rz2)) + 1

    # create assignment matrix
    asgn_matrix = np.zeros([length] * 2, dtype=np.int64)

    # iterate over all bin elements
    for element in range(length):
        for assign_to in range(length):
            # obtain the needed indices
            rz2_a_inds = np.argwhere(rz2 == assign_to)

            # get corresponding elements of rz1
            rz1_elements = rz1[rz2_a_inds]
            asgn_matrix[element, assign_to] = np.count_nonzero(rz1_elements != element)

    # solve using scipy
    r, permutation = linear_sum_assignment(asgn_matrix)

    # reduce to permutation
    cost = np.sum([asgn_matrix[i, permutation[i]] for i in range(length)])

    return cost, permutation
