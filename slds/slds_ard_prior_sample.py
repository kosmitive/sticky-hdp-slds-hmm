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
from scipy.stats import invwishart, matrix_normal


class StickyHDPSLDSHMMModelARDMixin:

    def slds_post_ard_dynamics_param_sampling(self):

        # split into reg problems
        prob_pt, prob_ct, n = self.__split_into_regression_problems()

        # perform one round of sampling
        self.__slds_post_ard_dynamics_sample_dyn_mat(prob_pt, prob_ct)
        self.__slds_post_ard_dynamics_sample_alpha()
        self.__slds_post_ard_dynamics_sample_cov_mat(prob_pt, prob_ct, n)

    def __slds_post_ard_dynamics_sample_cov_mat(self, prob_pt, prob_ct, n):

        # sample the dynamic matrices
        for k in range(self.hdp_prior.max_z):

            # compute sufficient statistics
            suff_stats = 0

            # obtain number of elements
            num_elements = prob_ct[k].shape[1]
            current_dyn = self.slds_params[0][k]

            # iterate over all elements
            for i in range(num_elements):
                v = prob_ct[k][:, i] - current_dyn @ prob_pt[k][:, i]
                suff_stats += np.outer(v, v)

            # sample covariance
            self.slds_params[1][k] = invwishart.rvs(n[k] + self.dyn_prior.n0,
                       suff_stats + self.dyn_prior.S0)

    def __slds_post_ard_dynamics_sample_dyn_mat(self, prob_pt, prob_ct):

        # obtain alphas
        alphas = self.slds_params[2]

        # sample the dynamic matrices
        for k in range(self.hdp_prior.max_z):

            # obtain number of elements
            num_elements = prob_ct[k].shape[1]

            # rename
            current_cov = self.slds_params[1][k]

            # iterate over
            inv_mean = np.zeros(self.ld ** 2)
            sigma0 = np.repeat(1 / alphas[k], self.ld)
            inv_cov = np.diag(sigma0)

            # iterate over all elements
            for i in range(num_elements):

                # construct matrix
                mat = np.tile(np.eye(self.ld), [1, self.ld])
                mat *= np.expand_dims(np.repeat(prob_pt[k][:, i], self.ld), 0)

                # add up
                inv_mean += mat.transpose() @ current_cov @ prob_ct[k][:, i]
                inv_cov += mat.transpose() @ current_cov @ mat

            # sample new matrix
            cov = np.linalg.inv(inv_cov)
            mean = cov @ inv_mean
            new_A = np.random.multivariate_normal(mean, cov)

            # sample new dynamic matrix
            self.slds_params[0][k] = new_A.reshape([self.ld] * 2)

    def __slds_post_ard_dynamics_sample_alpha(self):

        # sample the dynamic matrices
        for k in range(self.hdp_prior.max_z):

            # rename
            current_dyn = self.slds_params[0][k]
            alpha_a = self.dyn_prior.alpha0[0] + self.ld / 2
            alpha_b = self.dyn_prior.alpha0[1] + np.sum(current_dyn ** 2, 0)
            self.slds_params[2][k, :] = np.random.gamma(alpha_a, 1 / alpha_b)

    def __split_into_regression_problems(self):

        mode_seq = self.z
        pseudo_observations = self.x
        max_state = self.hdp_prior.max_z

        num_traces, num_steps, latent_dim = pseudo_observations.shape
        dim = latent_dim
        n = np.bincount(mode_seq[:, 0].reshape(-1), minlength=max_state)

        # calculate the bins and count for each bin how far it is progressed
        bins = np.bincount(mode_seq[:, 1:].reshape(-1), minlength=max_state)
        indices = np.zeros(max_state, dtype=np.int64)

        # reserve space for each bin
        prob_h = [np.empty([dim, b]) for b in bins]
        prob_oh = [np.empty([dim, b]) for b in bins]

        # iterate over all
        for d in range(num_traces):
            for t in range(1, num_steps):

                k = mode_seq[d, t]
                i = indices[k]
                prob_oh[k][:, i] = pseudo_observations[d, t]
                prob_h[k][:, i] = pseudo_observations[d, t-1]
                indices[k] += 1

        return prob_h, prob_oh, n + bins

