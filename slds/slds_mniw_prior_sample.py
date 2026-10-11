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

from slds.regression import split_regression_pairs


class StickyHDPSLDSHMMModelMNIWMixin:
    def slds_post_mniw_dynamics_param_sampling(self):

        state_dyn_mode = self.dynamics.A
        state_dyn_var_mode = self.dynamics.Sigma

        max_state = self.hdp_prior.max_z
        M = self.dyn_prior.M
        K = self.dyn_prior.K
        n0 = self.dyn_prior.n0
        S0 = self.dyn_prior.S0

        # split into reg problems
        prob_h, prob_oh, n = split_regression_pairs(self.z, self.x, self.hdp_prior.max_z)

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
            iw_scale = 0.5 * (iw_scale + iw_scale.T)
            state_dyn_var_mode[k] = invwishart.rvs(n0 + n[k], S0 + iw_scale)
            # MN(mean, left=Sigma, right=S_xx^{-1}). Passing the precisions
            # draws from a posterior that does not concentrate with the data.
            state_dyn_mode[k] = matrix_normal.rvs(
                new_m,
                state_dyn_var_mode[k],
                np.linalg.inv(s_hh),
            )
