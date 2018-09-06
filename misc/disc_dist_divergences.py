# MIT License
# 
# Copyright (c) 2018 Markus Semmler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import itertools

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
    l = np.maximum(np.max(rz1), np.max(rz2))+1

    # create assignment matrix
    asgn_matrix = np.zeros([l] * 2, dtype=np.int64)

    # iterate over all bin elements
    for element in range(l):
        for assign_to in range(l):

            # obtain the needed indices
            rz2_a_inds = np.argwhere(rz2 == assign_to)

            # get corresponding elements of rz1
            rz1_elements = rz1[rz2_a_inds]
            asgn_matrix[element, assign_to] = np.count_nonzero(rz1_elements != element)

    # solve using scipy
    r, permutation = linear_sum_assignment(asgn_matrix)

    # reduce to permutation
    cost = np.sum([asgn_matrix[i, permutation[i]] for i in range(l)])

    return cost, permutation