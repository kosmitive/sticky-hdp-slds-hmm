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


from os.path import join
import numpy as np
from scipy.stats import invwishart, multivariate_normal

from misc.disc_sample import sample_cat
from misc.hdd_save_load import save_json, load_json
from model.ard_prior import ARDPrior
from model.mniw_prior import MNIWPrior
from model.sticky_hdp_prior import StickyHDPPrior
from slds.slds_ard_prior_sample import StickyHDPSLDSHMMModelARDMixin

from slds.slds_block_sample_x import StickyHDPSLDSHMMModelBlockXMixin
from slds.slds_block_sample_z import StickyHDPSLDSHMMModelBlockZMixin
from slds.slds_init import StickyHDPSLDSHMMModelInitMixin
from slds.slds_kalman_filter_y import StickyHDPSLDSHMMModelKalmanYMixin
from slds.slds_mniw_prior_sample import StickyHDPSLDSHMMModelMNIWMixin
from slds.slds_sample_r import StickyHDPSLDSHMMModelEmissionMixin
from slds.slds_sample_sticky_hdp_params import StickyHDPSLDSHMMModelHDPMixin
from slds.slds_seq_sample_z import StickyHDPSLDSHMMModelSeqZMixin


class StickyHDPSLDSHMMModel(StickyHDPSLDSHMMModelBlockXMixin, StickyHDPSLDSHMMModelBlockZMixin,
                            StickyHDPSLDSHMMModelKalmanYMixin, StickyHDPSLDSHMMModelMNIWMixin,
                            StickyHDPSLDSHMMModelEmissionMixin, StickyHDPSLDSHMMModelHDPMixin,
                            StickyHDPSLDSHMMModelSeqZMixin, StickyHDPSLDSHMMModelInitMixin,
                            StickyHDPSLDSHMMModelARDMixin):

    fn_main = 'hdp_slds_hmm_model.json'
    fn_beta = 'beta.npy'
    fn_pi = 'pi.npy'
    fn_R0 = 'P0.npy'
    fn_ard_alpha = 'ard_alpha.npy'
    fn_P0 = 'R0.npy'
    fn_A = 'A.npy'
    fn_Sigma = 'Sigma.npy'
    fn_C = 'C.npy'
    fn_R = 'R.npy'
    fn_z = 'z.npy'
    fn_x = 'x.npy'
    fn_e = 'e.npy'
    fn_y = 'y.npy'
    fn_w = 'w.npy'

    def __init__(self, latent_dim, observ_dim, hdp_prior, dyn_prior, latent_prior, emission_prior, emission_proj):

        # sample some initial parameters
        self.dyn_prior = dyn_prior
        self.hdp_prior = hdp_prior
        self.latent_prior = latent_prior
        self.emission_prior = emission_prior
        self.ld = latent_dim
        self.od = observ_dim

        # sample
        self.slds_params = None
        self.hdp_params = list([None] * 5)
        self.C = emission_proj
        self.R = None
        self.z = None
        self.x = None
        self.e = None
        self.y = None
        self.w = None

        # reserve space for
        self.hist_alpha_kappa = list()
        self.hist_rho = list()
        self.hist_gamma = list()

        # dispatch
        if isinstance(dyn_prior, ARDPrior):
            self.dyn_prior_type = 'ard'
        elif isinstance(dyn_prior, MNIWPrior):
            self.dyn_prior_type = 'mniw'
        else:
            self.dyn_prior_type = None

    def calc_log_likelihood(self):
        """This method calculated the log likelihood of the model. Therefore all variables have to be set for it.

        :return: The negative log likelihood gets passed back in return.
        """

        # calc all terms separately
        state_transition_ll = self.__calc_mode_transition_ll()
        latent_transition_ll = self.__calc_latent_transition_ll()
        emission_ll = self.__calc_emission_ll()

        # pass back the negative sum
        return emission_ll + state_transition_ll + latent_transition_ll

    def __calc_latent_transition_ll(self):

        """Calculates log likelihood of the latent transitions."""

        # obtain transition matrices
        rsh_z = self.z[:, 1:].reshape(-1)
        latent_transition_ll = 0

        # calculate the term of the latent states
        num_modes, _ = self.hdp_params[4].shape
        num_traces, num_steps, latent_dim = self.x.shape
        comb_traces = self.x.reshape([num_traces * num_steps, -1])

        # iterate over all possible mode assignments
        for k in range(num_modes):
            # get indices of elements with k
            lin_indices = np.squeeze(np.argwhere(rsh_z == k), 1)

            # represents the transition matrix and the covariance
            tr_mat = self.slds_params[0][k]
            cov_mat = self.slds_params[1][k]

            # update indices
            lin_indices += np.floor(lin_indices / num_steps).astype(np.int64) + 1

            # calculate means
            mean_x = comb_traces[lin_indices - 1] @ tr_mat.transpose()

            # calc emission
            latent_transition_ll = 0
            for k in range(len(lin_indices)):
                latent_transition_ll += multivariate_normal.logpdf(comb_traces[lin_indices[k]], mean_x[k], cov_mat)

        return latent_transition_ll

    def __calc_mode_transition_ll(self):
        """Calculated the log likelihood of the state transitions."""

        num_modes = self.hdp_params[4].shape[0]

        # create cost matrix for all assignment combinations
        transitions = (num_modes * self.z[:, :-1] + self.z[:, 1:]).reshape(-1)
        transition_factors = np.bincount(transitions, minlength=num_modes ** 2)
        transition_count_matrix = transition_factors.reshape([num_modes] * 2)
        smoothed_measure = self.hdp_params[4] + 1e-6
        smoothed_measure = smoothed_measure / np.expand_dims(np.sum(smoothed_measure, 1), 0)
        state_transition_ll = np.sum(transition_count_matrix * np.log(smoothed_measure))

        if np.isnan(state_transition_ll):
            print("Test")

        return state_transition_ll

    def __calc_emission_ll(self):
        """Calculate log likelihood of emissions."""

        num_traces, num_steps, latent_dim = self.x.shape

        # first of all calculate the emission nll
        all_traces = num_traces * num_steps
        comb_traces = self.x.reshape([all_traces, -1])
        mean_y = comb_traces @ self.C.transpose()
        reshaped_y = self.y.reshape([all_traces, -1])

        # calc emission
        emission_ll = 0
        for k in range(all_traces):
            emission_ll += multivariate_normal.logpdf(reshaped_y[k], mean_y[k], self.R)

        return emission_ll

    def simulate(self, num_traces, num_steps):
        """Uses the prior information an so on, to simulate some traces.

        :return: collected traces
        """

        # reserve space for the simulation
        z = np.empty([num_traces, num_steps + 1], dtype=np.int64)
        x = np.empty([num_traces, num_steps + 1, self.ld])

        self.init_params()
        beta = self.hdp_params[3]
        pi = self.hdp_params[4]
        slds_params = self.slds_params

        # generate the emission noise before hand
        e = np.empty([num_traces, num_steps, self.ld])

        # repeat till all traces were generated
        for t in range(num_traces):

            # Simply choose one z from beta and draw a random multivariate x
            z[t, 0] = sample_cat(beta)
            x[t, 0] = np.random.multivariate_normal(np.zeros(self.ld), self.latent_prior)

            # execute all steps
            for s in range(1, num_steps + 1):

                # sample new mode
                pre_z = z[t, s - 1]
                cur_z = sample_cat(pi[pre_z])

                # get the current mode as an integer
                e[t, s - 1] = np.random.multivariate_normal(np.zeros(self.ld), slds_params[1][cur_z])
                x[t, s] = slds_params[0][cur_z] @ x[t, s - 1] + e[t, s - 1]
                z[t, s] = cur_z

        # emissions
        w = np.random.multivariate_normal(np.zeros(self.od), self.R, [num_traces * (num_steps + 1)])
        y = x.reshape([-1, self.ld]) @ self.C.transpose() + w
        y = y.reshape([num_traces, num_steps + 1, self.od])

        self.z = z[:, 1:]
        self.x = x[:, 1:]
        self.e = e
        self.y = y[:, 1:]
        self.w = w

    def init_params(self):

        # sample from prior
        slds_params = self.dyn_prior.prior_sample(self.hdp_prior.max_z)
        hdp_params = self.hdp_prior.max_likely_sample()

        # sample the covariance
        R = invwishart.rvs(self.emission_prior[0], self.emission_prior[1])

        self.R = R
        self.hdp_params = hdp_params
        self.slds_params = slds_params

    @staticmethod
    def load(folder):
        """Load the prior from the specified folder."""

        # reload data
        data = load_json(folder, StickyHDPSLDSHMMModel.fn_main)
        hdp_prior = StickyHDPPrior.load(folder)

        # depending on the type of prior
        dyn_prior = ARDPrior.load(folder) if data['prior'] == 'ard' else MNIWPrior.load(folder)

        # save prior matrices
        C = np.load(join(folder, StickyHDPSLDSHMMModel.fn_C))
        R0 = np.load(join(folder, StickyHDPSLDSHMMModel.fn_R0))
        latent_prior = np.load(join(folder, StickyHDPSLDSHMMModel.fn_P0))
        emission_prior = [data['r0'], R0]

        # create object
        ld = data['ld']
        od = data['od']
        m = StickyHDPSLDSHMMModel(ld, od, hdp_prior, dyn_prior, latent_prior, emission_prior, C)

        # load hdp
        beta = np.load(join(folder, StickyHDPSLDSHMMModel.fn_beta))
        pi = np.load(join(folder, StickyHDPSLDSHMMModel.fn_pi))
        m.hdp_params = list([data['alpha_kappa'], data['rho'], data['gamma'], beta, pi])

        # load dynamics
        A = np.load(join(folder, StickyHDPSLDSHMMModel.fn_A))
        Sigma = np.load(join(folder, StickyHDPSLDSHMMModel.fn_Sigma))
        m.slds_params = [A, Sigma]
        if data['prior'] is 'ard':
            m.slds_params.append(np.load(join(folder, StickyHDPSLDSHMMModel.fn_ard_alpha)))

        # save rest
        m.R = np.load(join(folder, StickyHDPSLDSHMMModel.fn_R))
        m.z = np.load(join(folder, StickyHDPSLDSHMMModel.fn_z))
        m.x = np.load(join(folder, StickyHDPSLDSHMMModel.fn_x))
        m.e = np.load(join(folder, StickyHDPSLDSHMMModel.fn_e))
        m.y = np.load(join(folder, StickyHDPSLDSHMMModel.fn_y))
        m.w = np.load(join(folder, StickyHDPSLDSHMMModel.fn_w))

        return m

    def store(self, folder):
        """Store the prior in the specified folder."""

        # store the dict
        prior_type = 'ard' if isinstance(self.dyn_prior, ARDPrior) else 'mniw'
        data = {
            'ld': self.ld,
            'od': self.od,
            'prior': prior_type,
            'r0': self.emission_prior[0],
            'alpha_kappa': self.hdp_params[0],
            'rho': self.hdp_params[1],
            'gamma': self.hdp_params[2]
        }
        save_json(folder, StickyHDPSLDSHMMModel.fn_main, data)

        # sample some initial parameters
        self.dyn_prior.store(folder)
        self.hdp_prior.store(folder)

        # save prior matrices
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_R0), self.emission_prior[1])
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_P0), self.latent_prior)

        # main parameters
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_A), self.slds_params[0])
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_Sigma), self.slds_params[1])

        if prior_type == 'ard':
            np.save(join(folder, StickyHDPSLDSHMMModel.fn_ard_alpha), self.slds_params[2])

        np.save(join(folder, StickyHDPSLDSHMMModel.fn_C), self.C)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_R), self.R)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_z), self.z)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_x), self.x)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_e), self.e)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_y), self.y)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_w), self.w)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_beta), self.hdp_params[3])
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_pi), self.hdp_params[4])
