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


class StickyHDPSLDSHMMModelSeqZMixin:

    def slds_post_seq_sample_z(self, forward_gain_mat, forward_gain_vec, backward_gain_mat, backward_gain_vec):
        """This method can be used to sequentially sample the mode assignment.

        :param conf: The learning
        :param mode_seq: The truncated mode sequence [DxT] of maximum L used modes.
        :param observations: The observation matrix [DxTxO] where each column is one time step.
        :return:
        """

        # get the maximum state
        max_state, _ = self.hdp_params[4].shape
        num_traces, num_steps, _ = self.y.shape
        state_dim, _ = self.slds_params[0][0].shape

        # iterate backwards
        for d in range(num_traces):
            for t in reversed(range(num_steps)):

                # shorthand
                z_t = self.z[d, t]
                inv_mat_gain = np.linalg.inv(forward_gain_mat[d, t])
                dyn_mat = self.slds_params[0][z_t]

                # calc the f values
                f = np.zeros(max_state, np.float64)
                for k in range(max_state):

                    # calc terms like in (4.27) 591409532-MIT.pdf
                    dyn_var_mat = self.slds_params[1][k]
                    b_val = dyn_mat @ inv_mat_gain
                    c_val = b_val @ dyn_mat.transpose()
                    inv_p_mat_k = dyn_var_mat + c_val
                    p_mat_k = np.linalg.inv(inv_p_mat_k)
                    p_vec_k = p_mat_k @ b_val @ forward_gain_vec[d, t]

                    # shorthand for the samoling of the discrete states
                    summed_mat = p_mat_k + backward_gain_mat[d, t+1]
                    inv_summed_mat = np.linalg.inv(summed_mat)
                    summed_vec = p_vec_k + backward_gain_vec[d, t+1]

                    # get inner distance
                    term = -0.5 * p_vec_k @ inv_p_mat_k @ p_vec_k + 0.5 * summed_vec @ inv_summed_mat @ summed_vec
                    f[k] = np.log(np.sqrt(np.linalg.det(p_mat_k) / np.log(np.linalg.det(summed_mat)) + 0.0001))
                    f[k] += term

                    # add the pi values if not at the corresponding end
                    if t > 1:
                        z_pt = self.z[d, t-1]
                        f[k] += np.log(self.hdp_params[4][z_pt, k] + 1e-6)

                    if t < num_steps - 1:
                        z_nt = self.z[d, t+1]
                        f[k] += np.log(self.hdp_params[4][k, z_nt] + 1e-6)

                e = np.exp(f - np.max(f))
                self.z[d, t] = np.random.choice(np.arange(max_state), p=e/np.sum(e))

