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
import scipy.stats as sp

from misc.disc_sample import sample_cat


class StickyHDPSLDSHMMModelBlockZMixin:

    def slds_post_block_sample_z(self):
        """ This method can be used to block sample the mode assignment, once

        :return: A vector [DxT] representing the sampled modes.
        """

        # calc messages
        gauss_dens = self.__calc_likelihood_transition()
        messages = self.__calc_messages(gauss_dens)

        # now sample the nodes
        self.__sample_from_messages(gauss_dens, messages)

    def __sample_from_messages(self, gauss_dens, messages):
        """This method samples from the passed gauss_dens, messages and pi as follows:

            z_t ~ sum_k(pi_{z_pt}(k) * gauss_dens[d,t,k] * messages[d,t,k])

        :param gauss_dens: The gaussian densities for the transition as [DxTxK] tensor.
        :param messages: The messages which were generated by message passing previously.
        :param pi: The transition probabilities between modes itself.
        :return: A tensor [DxT] containing for each trace and timestep one mode.
        """

        # get sp,e ma,es
        num_traces, num_steps, max_state = gauss_dens.shape
        pi = self.hdp_params[4]

        # iterate over and fill
        for t in range(num_steps):

            # simply
            f = pi[self.z[:, t - 1]] if t > 0 else np.ones([num_traces, max_state])
            f *= gauss_dens[:, t] * messages[:, t]

            for d in range(num_traces):
                z_t = np.random.choice(np.arange(max_state), p=f[d] / np.sum(f[d]))
                self.z[d, t] = z_t

    def __calc_messages(self, gauss_dens):
        """This method calculates the messages passed from the latent dynamic
        state x.

        :param gauss_dens: These are precomputed using calc_likelihood_transition_mode
        :param pi: The transition distribution between states.
        :return: A matrix [NxSxH] containing the messages for all possible combinations.
        """

        num_traces, num_steps, max_state = gauss_dens.shape

        # execute belief prog
        messages = np.ones([num_traces, num_steps + 1, max_state])

        # iterate over
        for t in reversed(range(num_steps)):
            for k in range(max_state):
                mult = gauss_dens[:, t] * messages[:, t + 1]
                res = np.einsum('kj,nj->nk', self.hdp_params[4], mult)
                messages[:, t, :] = res

        return messages

    def __calc_likelihood_transition(self, use_log=False):
        """This method calculated the likelihood per trace t, step s and mode k as follows:

            gauss_dens[t,s,k] = N(po[t], state_dyn_mode[k] @ po[t-1], state_dyn_var_mode[k])

        :param use_log: True, iff logits should be calculated.
        :return: A tensor res for which each (t,s,k) equals the likelihood of transition s in trace
                 t while in the k-th mode.
        """

        # remap some sizes
        num_traces, num_steps, _ = self.x.shape
        max_state, _, _ = self.slds_params[0].shape

        # reserve space and iteratively fill
        gauss_dens = np.ones([num_traces, num_steps, max_state])

        # calculate the gaussian densities
        for d in range(num_traces):
            for t in reversed(range(1, num_steps)):
                for k in range(max_state):
                    x = self.x[d, t]
                    mu = self.slds_params[0][k] @ self.x[d, t - 1]
                    fn = sp.multivariate_normal.logpdf if use_log else sp.multivariate_normal.pdf
                    gauss_dens[d, t, k] = fn(x, mu, self.slds_params[1][k])

        return gauss_dens