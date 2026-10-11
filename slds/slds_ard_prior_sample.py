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
from scipy.stats import invwishart

from slds.regression import split_regression_pairs


class StickyHDPSLDSHMMModelARDMixin:
    def slds_post_ard_dynamics_param_sampling(self):

        # split into reg problems
        prob_pt, prob_ct, n = split_regression_pairs(self.z, self.x, self.hdp_prior.max_z)

        # perform one round of sampling
        self.__slds_post_ard_dynamics_sample_dyn_mat(prob_pt, prob_ct)
        self.__slds_post_ard_dynamics_sample_alpha()
        self.__slds_post_ard_dynamics_sample_cov_mat(prob_pt, prob_ct, n)

    def __slds_post_ard_dynamics_sample_cov_mat(self, prob_pt, prob_ct, n):

        # sample the dynamic matrices
        for k in range(self.hdp_prior.max_z):
            current_dyn = self.dynamics.A[k]
            if prob_ct[k].shape[1] == 0:
                suff_stats = np.zeros((self.ld, self.ld))
            else:
                residual = prob_ct[k] - current_dyn @ prob_pt[k]
                suff_stats = residual @ residual.T
            self.dynamics.Sigma[k] = invwishart.rvs(
                n[k] + self.dyn_prior.n0,
                suff_stats + self.dyn_prior.S0,
            )

    def __slds_post_ard_dynamics_sample_dyn_mat(self, prob_pt, prob_ct):

        # obtain alphas
        alphas = self.dynamics.alpha

        # sample the dynamic matrices
        for k in range(self.hdp_prior.max_z):
            # Eq. (30): information form uses Sigma^{-1}, and the ARD prior
            # precision on column j is alpha_j (n replicates under column-major vec).
            noise_precision = np.linalg.inv(self.dynamics.Sigma[k])
            info, precision = _ard_normal_equations(
                prob_pt[k], prob_ct[k], noise_precision, alphas[k]
            )

            cov = np.linalg.inv(precision)
            cov = 0.5 * (cov + cov.T)
            mean = cov @ info
            new_A = np.random.multivariate_normal(mean, cov)
            # Column-major vectorization matches the design matrix above.
            self.dynamics.A[k] = new_A.reshape((self.ld, self.ld), order="F")

    def __slds_post_ard_dynamics_sample_alpha(self):

        # sample the dynamic matrices
        for k in range(self.hdp_prior.max_z):
            # rename
            current_dyn = self.dynamics.A[k]
            # Eq. (31): rate += (1/2) * sum of squares in the column.
            alpha_a = self.dyn_prior.alpha0[0] + self.ld / 2
            alpha_b = self.dyn_prior.alpha0[1] + 0.5 * np.sum(current_dyn**2, axis=0)
            self.dynamics.alpha[k, :] = np.random.gamma(alpha_a, 1.0 / alpha_b)


def _ard_normal_equations(parents, children, noise_precision, column_precision):
    """Column-major normal equations for one mode.

    The design row used by the original loop is ``kron(parent, I)``. Block
    ``(j, k)`` of the Gram matrix is ``(parent_j parent_k) * Sigma^{-1}``.

    Args:
        parents: Parent states, shape ``(latent, count)``.
        children: Child states, shape ``(latent, count)``.
        noise_precision: ``Sigma^{-1}``.
        column_precision: ARD precision for each column of ``A``.

    Returns:
        The information vector and the precision matrix of ``vec(A)``.
    """

    latent_dim = parents.shape[0]
    info = np.zeros(latent_dim**2)
    precision = np.diag(np.repeat(column_precision, latent_dim))
    if parents.shape[1] == 0:
        return info, precision
    weighted = noise_precision @ children
    gram = parents @ parents.T
    for column in range(latent_dim):
        info[column * latent_dim : (column + 1) * latent_dim] = weighted @ parents[column]
        for row in range(latent_dim):
            block = slice(column * latent_dim, (column + 1) * latent_dim)
            other = slice(row * latent_dim, (row + 1) * latent_dim)
            precision[block, other] += gram[column, row] * noise_precision
    return info, precision
