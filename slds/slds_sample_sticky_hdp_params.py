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

from misc.disc_sequences import count_state_transitions
from scipy.stats import bernoulli, beta


class StickyHDPSLDSHMMModelHDPMixin:

    def calc_transitions(self):

        self.transitions = count_state_transitions(
            self.z,
            self.hdp_prior.max_z
        )

    def sample_post_auxiliary_vars(self):

        beta = self.hdp_params[3]
        alpha_kappa = self.hdp_params[0]
        rho = self.hdp_params[1]

        # calc
        kappa = rho * alpha_kappa
        alpha = (1 - rho) * alpha_kappa

        transitions = self.transitions

        trunc_level = np.size(beta)
        m = np.zeros([trunc_level] * 2)# + 0.0001

        for j in range(trunc_level):
            for k in range(trunc_level):
                term = alpha * beta[k] + np.where(j == k, kappa, 0)
                offsets = np.arange(transitions[j, k]) + 1
                prob = term / (offsets + term)
                x = [np.random.binomial(1, p) for p in prob]
                m[j, k] += np.count_nonzero(x)

        # get rho
        rho = kappa / (alpha + kappa)
        p = rho / (rho + beta * (1 - rho))

        # now simply create
        w = np.asarray([np.random.binomial(m[j, j], p[j]) for j in range(trunc_level)])
        m_hat = m - np.diag(w)

        # save auxiliary vars
        self.m = m
        self.m_hat = m_hat
        self.w = w

    def sample_post_hdp_rho(self):

        m = self.m
        w = self.w
        prior = self.hdp_prior.rho0

        sw = np.sum(w)
        a = sw + np.asarray(prior[0])
        b = np.sum(m) - sw + prior[1]

        # sample rho
        rho = np.random.beta(a, b)
        self.hist_rho.append((a, b))
        self.hdp_params[1] = rho

    def sample_post_hdp_gamma(self):

        m_hat = self.m_hat
        prior = self.hdp_prior.gamma0
        gamma = self.hdp_params[2]

        # first of all sample auxiliary vars
        bigk = np.count_nonzero(np.greater(np.sum(m_hat, axis=0), 0))

        # and another one
        summed_m_hat = np.sum(m_hat)
        eta = np.random.beta(gamma + 1, summed_m_hat + 1e-6)

        # sample gamma
        fp = prior[0] + bigk
        sp = prior[1] - np.log(eta)
        p_mixing = summed_m_hat / (summed_m_hat + gamma)
        mixing = np.random.binomial(n=1, p=p_mixing)

        # now sample gamma
        a = fp - mixing
        b = 1 / sp
        gamma = np.random.gamma(a, b)
        self.hist_gamma.append((a, b))
        self.hdp_params[2] = gamma

    def sample_post_hdp_alpha_kappa(self):

        transitions = self.transitions
        prior = self.hdp_prior.alpha_kappa0
        alpha_kappa = self.hdp_params[0]

        # set trunk level
        trunc_level = transitions.shape[0]

        # sample again auxiliary vars
        n_rows = np.sum(transitions, axis=1)

        # sample auxiliary r and s
        r = [beta.rvs(alpha_kappa + 1, n_rows[j] + 1e-6) for j in range(trunc_level)]
        s = [bernoulli.rvs(n_rows[j] / (n_rows[j] + alpha_kappa)) for j in range(trunc_level)]

        # sample the sum of alpha and kappa
        summed_s = np.sum(s)
        summed_r = np.sum(np.log(r))

        # get a and b
        a = prior[0] + np.sum(self.m) - summed_s
        b = 1 / (prior[1] - summed_r)
        alpha_kappa = np.random.gamma(a, b)

        self.hist_alpha_kappa.append((a, b))
        self.hdp_params[0] = alpha_kappa

    def sample_post_hdp_pi(self):

        transitions = self.transitions
        beta = self.hdp_params[3]
        alpha_kappa = self.hdp_params[0]
        rho = self.hdp_params[1]

        # calc
        kappa = rho * alpha_kappa
        alpha = (1 - rho) * alpha_kappa

        trunc_level = transitions.shape[0]
        pi = np.empty(2 * [trunc_level], np.float64)

        # update pi
        id = np.eye(trunc_level)
        for k in range(trunc_level):
            probs = alpha * beta + transitions[k] + id[k] * kappa
            pi[k] = np.random.dirichlet(probs)

        self.hdp_params[4] = pi

    def sample_post_hdp_beta(self):

        trunc_level = self.hdp_prior.max_z
        summed_m_hat = np.sum(self.m_hat, axis=0)
        self.hdp_params[3] = np.random.dirichlet(self.hdp_params[2] / trunc_level + summed_m_hat)