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


import os

import numpy as np
from os.path import join

from misc.ansi_color_codes import ACC
from misc.disc_dist_divergences import minimum_hamming_distance, kl_divergence
from misc.history import History
from misc.output import print_line, print_array, print_trace, gen_plot
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel


class StickyHDPSLDSHMMConfiguration:

    fn_main = 'gt_sticky_hdp_slds_hmm_model.json'

    def __init__(self, gt=None, it=None):

        self.best_it = None
        self.gt = gt
        self.it = it

        # sample control variables
        self.sample_x = None
        self.sample_block_z = None
        self.sample_seq_z = None
        self.sample_hdp_beta = None
        self.sample_hdp_alpha_kappa = None
        self.sample_hdp_rho = None
        self.sample_hdp_pi = None
        self.sample_hdp_gamma = None
        self.sample_dynamics = None
        self.sample_r = None

        # rounds
        self.rounds_ard = 1
        self.rounds_hdp = 1
        self.rounds_hmm = 1

        # nll for gt
        self.gt_ll = None
        self.it_ll = None
        self.best_ll = None
        self.store_hdp_hist = False

        # create history
        self.history = None
        self.log_step = 0

    @staticmethod
    def init_for_test(test_conf):

        print_line()

        print("Create Model")
        gt = StickyHDPSLDSHMMModel.load(test_conf['model_path'])
        it = StickyHDPSLDSHMMModel(gt.ld,
                                   gt.od,
                                   gt.hdp_prior,
                                   gt.dyn_prior,
                                   gt.latent_prior,
                                   gt.emission_prior,
                                   gt.C)

        # create model and write which values should be sampled
        m = StickyHDPSLDSHMMConfiguration(gt=gt, it=it)
        m.gt_ll = gt.calc_log_likelihood()

        print("Initializing simulation.")
        # some renaming
        num_traces, num_steps, _ = m.gt.y.shape
        max_z = m.gt.hdp_prior.max_z

        # rename
        m.sample_x = test_conf['sample_x']
        m.sample_block_z = test_conf['sample_block_z']
        m.sample_seq_z = test_conf['sample_seq_z']
        m.sample_hdp_beta = test_conf['sample_hdp_beta']
        m.sample_hdp_alpha_kappa = test_conf['sample_hdp_alpha_kappa']
        m.sample_hdp_rho = test_conf['sample_hdp_rho']
        m.sample_hdp_pi = test_conf['sample_hdp_pi']
        m.sample_hdp_gamma = test_conf['sample_hdp_gamma']
        m.sample_dynamics = test_conf['sample_dynamics']
        m.sample_r = test_conf['sample_r']

        # rounds
        if 'rounds_ard' in test_conf: m.rounds_ard = test_conf['rounds_ard']
        if 'rounds_hdp' in test_conf: m.rounds_ard = test_conf['rounds_hdp']
        if 'rounds_hmm' in test_conf: m.rounds_hmm = test_conf['rounds_hmm']

        # store hdp params
        m.store_hdp_hist = test_conf['store_hdp_hist'] if 'store_hdp_hist' in test_conf else False

        # copy some information depending on the above set information
        m.it.y = m.gt.y
        m.it.init_params()
        m.it.slds_init_z()

        # create history
        m.history = History(test_conf['gibbs_steps'])
        m.history.add_cat('ll')
        m.history.add_cat('')

        # copy over
        if not m.sample_x: m.it.x = m.gt.x
        else: m.history.add_cat("mean_error_x")

        if not m.sample_block_z and not m.sample_seq_z: m.it.z = m.gt.z
        else: m.history.add_cat("mean_ham_dist_z")

        if not m.sample_hdp_beta: m.it.hdp_params[3] = m.gt.hdp_params[3]
        else:
            m.history.add_cat("i_projection_beta")
            m.history.add_cat("m_projection_beta")

        if not m.sample_hdp_alpha_kappa: m.it.hdp_params[0] = m.gt.hdp_params[0]
        else: m.history.add_cat("abs_error_alpha_kappa")

        if not m.sample_hdp_rho: m.it.hdp_params[1] = m.gt.hdp_params[1]
        else: m.history.add_cat("abs_error_rho")

        if not m.sample_hdp_pi: m.it.hdp_params[4] = m.gt.hdp_params[4]
        else:
            m.history.add_cat("i_projection_pi")
            m.history.add_cat("m_projection_pi")

        if not m.sample_hdp_gamma: m.it.hdp_params[2] = m.gt.hdp_params[2]
        else: m.history.add_cat("abs_error_gamma")

        if not m.sample_dynamics: m.it.slds_params = m.gt.slds_params
        else:
            for k in range(max_z):
                m.history.add_cat("frob_norm_dyn_{}".format(k))
                m.history.add_cat("frob_norm_cov_{}".format(k))

        if not m.sample_r: m.it.R = m.gt.R
        else: m.history.add_cat("frob_norm_r")

        # pass back model
        return m

    def log_errors(self):

        # log the model after the iteration
        max_z = self.gt.hdp_prior.max_z
        nll = self.it_ll
        self.history.add_value('ll', self.log_step, nll)
        num_traces, num_steps, _ = self.gt.y.shape
        divi = num_traces * num_steps

        # when both should be sampled.
        if self.sample_x:
            err = np.sum(np.abs(self.gt.x - self.it.x))
            self.history.add_value('mean_error_x', self.log_step, err / divi)

        if self.sample_seq_z or self.sample_block_z or self.sample_dynamics:
            cost_z, perm_z = minimum_hamming_distance(self.it.z, self.gt.z)

            if self.sample_seq_z or self.sample_block_z:
                self.history.add_value('mean_ham_dist_z', self.log_step, (cost_z / divi))

            if self.sample_dynamics:
                for k in range(max_z):
                    dist_dyn = np.linalg.norm(self.it.slds_params[0][k] - self.gt.slds_params[0][perm_z[k]], ord='fro')
                    dist_cov = np.linalg.norm(self.it.slds_params[1][k] - self.gt.slds_params[1][perm_z[k]], ord='fro')
                    self.history.add_value("frob_norm_dyn_{}".format(k), self.log_step, dist_dyn)
                    self.history.add_value("frob_norm_cov_{}".format(k), self.log_step, dist_cov)

        # when both should be sampled.
        if self.sample_hdp_alpha_kappa:
            err = np.abs(self.gt.hdp_params[0] - self.it.hdp_params[0])
            self.history.add_value('abs_error_alpha_kappa', self.log_step, err)

        # when both should be sampled.
        if self.sample_hdp_rho:
            err = np.abs(self.gt.hdp_params[1] - self.it.hdp_params[1])
            self.history.add_value('abs_error_rho', self.log_step, err)

        # when both should be sampled.
        if self.sample_hdp_gamma:
            err = np.abs(self.gt.hdp_params[2] - self.it.hdp_params[2])
            self.history.add_value('abs_error_gamma', self.log_step, err)

        # when both should be sampled.
        if self.sample_hdp_beta:
            err = kl_divergence(self.it.hdp_params[3], self.gt.hdp_params[3])
            self.history.add_value('i_projection_beta', self.log_step, err)

            err = kl_divergence(self.gt.hdp_params[3], self.it.hdp_params[3])
            self.history.add_value('m_projection_beta', self.log_step, err)

        if self.sample_hdp_pi:
            err = kl_divergence(self.it.hdp_params[4], self.gt.hdp_params[4])
            self.history.add_value('i_projection_pi', self.log_step, err)

            err = kl_divergence(self.gt.hdp_params[4], self.it.hdp_params[4])
            self.history.add_value('m_projection_pi', self.log_step, err)

        if self.sample_r:
            err = np.linalg.norm(self.it.R - self.gt.R, ord='fro')
            self.history.add_value('frob_norm_r', self.log_step, err)

        # increase step by one
        self.log_step += 1

    def print_model(self, print_out_prior=True, num_traces=1):

        num_modes, _ = self.it.hdp_params[4].shape

        if print_out_prior:

            # simply print out hte first trace

            print_line()
            if np.isnan(self.it_ll):
                print("Test")
            print("\t| ll = \t\t\t{:.5f} \t{}[{:.5f}]{}\t{}[{:.5f}]{}".format(self.it_ll, ACC.OkBlue, self.gt_ll, ACC.End, ACC.Red, self.best_ll, ACC.End))
            ham, _ = minimum_hamming_distance(self.it.z, self.gt.z)
            print("\t| ham_dist = \t{}".format(ham))
            print_line()
            print("\t| alpha_kappa = \t{:.5f} \t{}[{:.5f}]{}".format(self.it.hdp_params[0], ACC.OkBlue, self.gt.hdp_params[0], ACC.End))
            print("\t| rho = \t\t\t{:.5f} \t{}[{:.5f}]{}".format(self.it.hdp_params[1], ACC.OkBlue, self.gt.hdp_params[1], ACC.End))
            print("\t| gamma = \t\t\t{:.5f} \t{}[{:.5f}]{}".format(self.it.hdp_params[2], ACC.OkBlue, self.gt.hdp_params[2], ACC.End))
            print_line()
            print("\t| beta = \t[", *print_array(self.it.hdp_params[3]), "] \t", ACC.OkBlue, "[", *print_array(self.gt.hdp_params[3]), "]", ACC.End)
            print_line()
            print("\t| pi = \t\t[", *print_array(self.it.hdp_params[4][0]), "] \t", ACC.OkBlue, "[", *print_array(self.gt.hdp_params[4][0]), "]", ACC.End)
            for i in range(1, num_modes):
                print("\t\t\t\t[", *print_array(self.it.hdp_params[4][i]), "] \t", ACC.OkBlue, "[", *print_array(self.gt.hdp_params[4][i]), "]", ACC.End)

        # print out traces
        for i in range(num_traces):
            print_trace(i, self.it.z, self.it.x, self.it.y, self.gt.z, self.gt.x, self.gt.y)

    def store_run(self, folder):

        # save the plots
        gen_plot(self.history.mem['ll'], os.path.join(folder, 'll.pdf'), 'll')

        if self.sample_x:
            gen_plot(self.history.mem['mean_error_x'], os.path.join(folder, 'mean_abs_error_x.pdf'), 'mean_abs_error_x')

        if self.sample_seq_z or self.sample_block_z:
            gen_plot(self.history.mem['mean_ham_dist_z'], os.path.join(folder, 'mean_ham_dist_z.pdf'), 'mean_ham_dist_z')

        # when both should be sampled.
        if self.sample_hdp_alpha_kappa:
            gen_plot(self.history.mem['abs_error_alpha_kappa'], os.path.join(folder, 'abs_error_alpha_kappa.pdf'), 'abs_error_alpha_kappa')

        if self.sample_hdp_rho:
            gen_plot(self.history.mem['abs_error_rho'], os.path.join(folder, 'abs_error_rho.pdf'), 'abs_error_rho')

        if self.sample_hdp_gamma:
            gen_plot(self.history.mem['abs_error_gamma'], os.path.join(folder, 'abs_error_gamma.pdf'), 'abs_error_gamma')

        if self.sample_hdp_pi:
            gen_plot(self.history.mem['i_projection_pi'], os.path.join(folder, 'i_projection_pi.pdf'), 'i_projection_pi')
            gen_plot(self.history.mem['m_projection_pi'], os.path.join(folder, 'm_projection_pi.pdf'), 'm_projection_pi')

        if self.sample_hdp_beta:
            gen_plot(self.history.mem['i_projection_beta'], os.path.join(folder, 'i_projection_beta.pdf'), 'i_projection_beta')
            gen_plot(self.history.mem['m_projection_beta'], os.path.join(folder, 'm_projection_beta.pdf'), 'm_projection_beta')

        if self.sample_dynamics:
            v = [self.history.mem['frob_norm_dyn_{}'.format(k)] for k in range(1, self.gt.hdp_prior.max_z)]
            gen_plot(v, os.path.join(folder, 'frob_norm_dyn.pdf'), 'frob_norm_dyn')

            v = [self.history.mem['frob_norm_cov_{}'.format(k)] for k in range(1, self.gt.hdp_prior.max_z)]
            gen_plot(v, os.path.join(folder, 'frob_norm_cov.pdf'), 'frob_norm_cov')

        if self.sample_r:
            gen_plot(self.history.mem['frob_norm_r'], os.path.join(folder, 'frob_norm_r.pdf'), 'frob_norm_r')

        # save history
        self.history.save(folder)

        # save hdp hist if necessary
        if self.store_hdp_hist:
            np.savetxt(os.path.join(folder, 'hist_rho.txt'), self.it.hist_rho)
            np.savetxt(os.path.join(folder, 'hist_alpha_kappa.txt'), self.it.hist_alpha_kappa)
            np.savetxt(os.path.join(folder, 'hist_gamma.txt'), self.it.hist_gamma)

    def gibbs_step(self):

        # if x or z has to be sampled calc kalman gains
        backward_gains = None

        # update alpha and kappa
        for k in range(self.rounds_hmm):

            if self.sample_x or self.sample_seq_z:
                backward_gains = self.it.slds_backward_kalman_filter_gains()

            if self.sample_seq_z:

                # get all kalman gains
                forward_gains = self.it.slds_forward_kalman_filter_gains()
                self.it.slds_post_seq_sample_z(forward_gains[0], forward_gains[1], backward_gains[0], backward_gains[1])

            # first of all sample
            if self.sample_x: self.it.slds_post_block_sample_x(backward_gains[0], backward_gains[1])

        if self.sample_block_z: self.it.slds_post_block_sample_z()

        # update alpha and kappa
        for k in range(self.rounds_hdp):

            # get auxiliary vars if necessary
            if self.sample_hdp_beta or self.sample_hdp_alpha_kappa or self.sample_hdp_gamma or self.sample_hdp_pi or self.sample_hdp_rho:
                self.it.calc_transitions()

            # get auxiliary vars if necessary
            if self.sample_hdp_beta or self.sample_hdp_alpha_kappa or self.sample_hdp_gamma or self.sample_hdp_rho:
                self.it.sample_post_auxiliary_vars()

            if self.sample_hdp_alpha_kappa: self.it.sample_post_hdp_alpha_kappa()
            if self.sample_hdp_rho: self.it.sample_post_hdp_rho()
            if self.sample_hdp_gamma: self.it.sample_post_hdp_gamma()
            if self.sample_hdp_beta: self.it.sample_post_hdp_beta()
            if self.sample_hdp_pi: self.it.sample_post_hdp_pi()

        # sample dynamics and R
        if self.sample_dynamics:
            if self.it.dyn_prior_type is 'ard':
                for k in range(self.rounds_ard):
                    self.it.slds_post_ard_dynamics_param_sampling()
            else:
                self.it.slds_post_mniw_dynamics_param_sampling()

        if self.sample_r: self.it.slds_post_sample_r()

        # save and store log likelihood
        self.it_ll = self.it.calc_log_likelihood()

        # save if log likelihood is better than the previous known one
        if self.best_ll is None or self.it_ll > self.best_ll:
            self.best_ll = self.it_ll

    @staticmethod
    def load(folder):
        """Load the prior from the specified folder."""

        # obtain the paths
        gt_path = os.path.join(folder, "gt")
        it_path = os.path.join(folder, "it")
        m = StickyHDPSLDSHMMConfiguration()
        m.gt = StickyHDPSLDSHMMModel.load(gt_path)
        m.it = StickyHDPSLDSHMMModel.load(it_path)

        return m

    def store(self, folder):
        """Store the prior in the specified folder."""

        gt_path = os.path.join(folder, "gt")
        it_path = os.path.join(folder, "it")
        self.gt.store(gt_path)
        self.it.store(it_path)