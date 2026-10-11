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


class StickyHDPSLDSHMMModelSeqZMixin:

    def slds_post_seq_sample_z(
        self,
        forward_gain_mat: np.ndarray,
        forward_gain_vec: np.ndarray,
        backward_gain_mat: np.ndarray,
        backward_gain_vec: np.ndarray,
    ) -> None:
        """Sample each mode given the others, marginalizing the latent state.

        The marginal likelihood is Eq. (39) of Fox et al. (arXiv:1003.3829).
        ``A^{(k)}`` and ``Sigma^{(k)}`` are both taken from the candidate mode.
        The first mode is scored under ``beta``; every later mode uses ``pi``.
        """

        max_state, _ = self.hdp_params[4].shape
        num_traces, num_steps, _ = self.y.shape

        # iterate backwards
        beta = self.hdp_params[3]
        pi = self.hdp_params[4]
        for d in range(num_traces):
            for t in reversed(range(num_steps)):

                # Forward information at this index is about x_{t-1} and does
                # not depend on the candidate mode.
                inv_mat_gain = np.linalg.inv(forward_gain_mat[d, t])
                f = np.zeros(max_state, np.float64)
                for k in range(max_state):

                    # Eq. (39)–(40): both A and Sigma belong to the candidate mode.
                    dyn_mat = self.slds_params[0][k]
                    dyn_var_mat = self.slds_params[1][k]
                    b_val = dyn_mat @ inv_mat_gain
                    c_val = b_val @ dyn_mat.transpose()
                    inv_p_mat_k = dyn_var_mat + c_val
                    p_mat_k = np.linalg.inv(inv_p_mat_k)
                    p_vec_k = p_mat_k @ b_val @ forward_gain_vec[d, t]

                    summed_mat = p_mat_k + backward_gain_mat[d, t + 1]
                    inv_summed_mat = np.linalg.inv(summed_mat)
                    summed_vec = p_vec_k + backward_gain_vec[d, t + 1]

                    # log |Λ|^{1/2} / |Λ + Λ^b|^{1/2}, plus the quadratic terms.
                    # The determinant ratio is not wrapped in an extra log.
                    sign_p, logdet_p = np.linalg.slogdet(p_mat_k)
                    sign_s, logdet_s = np.linalg.slogdet(summed_mat)
                    term = -0.5 * p_vec_k @ inv_p_mat_k @ p_vec_k
                    term += 0.5 * summed_vec @ inv_summed_mat @ summed_vec
                    if sign_p <= 0 or sign_s <= 0:
                        f[k] = -np.inf
                    else:
                        f[k] = 0.5 * (logdet_p - logdet_s) + term

                    if t > 0:
                        z_pt = self.z[d, t - 1]
                        f[k] += np.log(pi[z_pt, k] + 1e-12)
                    else:
                        f[k] += np.log(beta[k] + 1e-12)

                    if t < num_steps - 1:
                        z_nt = self.z[d, t + 1]
                        f[k] += np.log(pi[k, z_nt] + 1e-12)

                finite = np.isfinite(f)
                if not np.any(finite):
                    probs = np.full(max_state, 1.0 / max_state)
                else:
                    f = np.where(finite, f, -np.inf)
                    shifted = np.exp(f - np.max(f[finite]))
                    probs = shifted / np.sum(shifted)
                self.z[d, t] = np.random.choice(np.arange(max_state), p=probs)

