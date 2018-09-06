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


import numpy as np
from misc.disc_sample import sample_cat


def stick_breaking_construction(gamma, k_max=10000):
    """Represents a simple stick breaking construction GEM(gamma). It gets constructed as follows:

                k = 1, ..., k_max
                v_k ~ Beta(1, alpha)
                beta_k = v_k * prod(i=1, i < k, v_i)

    We can then write:

                beta ~ stick_breaking_construction(alpha) = GEM(gamma)

    :param gamma: The gamma value of the stick-breaking construction
    :param k_max: The number of drawn samples for the stick breaking construction usually 10000
    :return: The generated unnormalized distribution.
    """

    # some base samples
    v = np.random.beta(1, gamma, k_max)
    r = np.cumprod(1 - v[:-1])
    mid = np.concatenate(([1], r))
    beta = v * mid

    return beta


def stick_breaking_construction_dp(alpha, base_measure, num_modes, k_max=10000):
    """Represents a simple stick breaking construction of a dirichlet process DP(alpha, base_measure).
    It gets constructed like:

        beta ~ GEM(alpha)
        sigma_k ~ base_measure

    It outputs the measure sum_k(beta_k * delta(sigma_k))

    :param alpha: The alpha value of the dirichlet process.
    :param base_measure: This is a a vector, containing a measure over discrete values until the max mode.
    :param num_modes: The number of different modes
    :param k_max: The number of drawn samples for the stick breaking construction usually 10000
    :return: The drawn dirichlet process distribution.
    """

    # replace base measure if it is a string
    if base_measure == 'uniform':
        base_measure = np.ones(num_modes) / num_modes

    # generate some samples
    base_samples = np.random.choice(num_modes, p=base_measure, size=k_max)
    beta = stick_breaking_construction(alpha, k_max)

    # create a m beta
    m_beta = np.zeros(num_modes)
    for m in range(num_modes):
        m_beta[m] = np.sum(np.where(base_samples == m, beta, 0))

    return m_beta / np.sum(m_beta)


def stick_breaking_construction_sticky_hdp(alpha, kappa, gamma, num_modes, base_measure='uniform', k_max=10000):
    """Creates a stick-breaking construction of a sticky hierarchical dirichlet prior. It
    basically uses kappa to add some self-excitement to each state, while alpha and gamma
    control the transition finally chosen. It is chosen as:

                beta ~ GEM(gamma)
                pi_k ~ DP(alpha + kappa, alpha * beta + kappa * e_k)

    :param alpha: The amount of sticking to the
    :param kappa: The amount of self excitement, is for each state the same.
    :param gamma: The b parameter of the beta of the inner stick breaking construction.
    :param num_modes: The number of different modes
    :param k_max: The number of drawn samples for the stick breaking construction usually 10000
    :return:
    """

    # sample from beta distribution
    beta = stick_breaking_construction_dp(gamma, base_measure, num_modes, k_max)

    # sample pre pi distribution
    pi = np.empty([num_modes, num_modes], dtype=np.float64)

    # for all modes sample a distribution with self excitement.
    for m in range(num_modes):

        # get specific base measure
        ab = alpha * beta
        ab[m] += kappa
        ab /= alpha + kappa
        ak = alpha + kappa

        # again stick breaking construction
        pi[m] = stick_breaking_construction_dp(ak, ab, num_modes, k_max)

    return beta, pi
