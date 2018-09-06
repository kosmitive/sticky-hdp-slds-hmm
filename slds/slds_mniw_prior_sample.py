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


class StickyHDPSLDSHMMModelMNIWMixin:

    def slds_post_mniw_dynamics_param_sampling(self):

        state_dyn_mode = self.slds_params[0]
        state_dyn_var_mode = self.slds_params[1]

        max_state = self.hdp_prior.max_z
        M = self.dyn_prior.M
        K = self.dyn_prior.K
        n0 = self.dyn_prior.n0
        S0 = self.dyn_prior.S0

        # split into reg problems
        prob_h, prob_oh, n = self.__split_into_regression_problems()

        # short hands
        MK = M @ K
        MKM = M @ K @ M.transpose()

        # calc suff stats
        for k in range(max_state):

            # sample the new hyper parameters
            s_hh = prob_h[k] @ prob_h[k].transpose() + K
            s_oh = prob_oh[k] @ prob_h[k].transpose() + MK
            s_oo = prob_oh[k] @ prob_oh[k].transpose() + MKM

            new_m = s_oh @ np.linalg.inv(s_hh)
            iw_scale = s_oo - new_m @ s_oh.transpose()
            state_dyn_var_mode[k] = invwishart.rvs(n0 + n[k], S0 + iw_scale)
            state_dyn_mode[k] = matrix_normal.rvs(new_m, np.linalg.inv(state_dyn_var_mode[k]), s_hh)

        self.slds_params = [state_dyn_mode, state_dyn_var_mode]

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

