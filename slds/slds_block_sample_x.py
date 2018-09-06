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


class StickyHDPSLDSHMMModelBlockXMixin:

    def slds_post_block_sample_x(self, backward_gain_mat, backward_gain_vec):

        num_traces, num_steps, state_dim = backward_gain_vec.shape
        num_steps -= 1

        pseudo_observations = np.empty([num_traces, num_steps + 1, state_dim])
        pseudo_observations[:, 0] = np.random.multivariate_normal(np.zeros(state_dim), np.eye(state_dim), num_traces)

        for d in range(num_traces):

            # by forward iter
            for t in range(num_steps):
                z_t = self.z[d, t]
                dyn_mat = self.slds_params[0][z_t]
                dyn_var_mat = self.slds_params[1][z_t]
                inv_dyn_var_mat = np.linalg.inv(dyn_var_mat)

                inv_sigma_lam = np.linalg.inv(inv_dyn_var_mat + backward_gain_mat[d, t + 1])
                middle = inv_dyn_var_mat @ dyn_mat @ pseudo_observations[d, t] + backward_gain_vec[d, t + 1]
                pseudo_observations[d, t + 1] = np.random.multivariate_normal(inv_sigma_lam @ middle, inv_sigma_lam)

        self.x = pseudo_observations[:, 1:]