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

from data.generate_hdp_slds_groundtruth import generate_hdp_slds_groundtrouth
from misc.output import print_line
from model.ard_prior import ARDPrior
from model.mniw_prior import MNIWPrior
from model.sticky_hdp_prior import StickyHDPPrior
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel


def generate_mniw_data(root='./data_packs', name='hdp_slds_mniw'):

    print_line()
    print_line("Generate HDP-SLDS-HMM using MNIW prior")

    # some general settings
    num_traces = 50
    num_steps = 100
    ld = 3
    od = 2
    max_z = 5

    # projection
    C = np.hstack((np.eye(od), np.zeros([od, ld-od])))

    # setup hdp prior
    alpha_kappa0 = [15.0, 3.0]
    rho0 = [5.0, 2.0]
    gamma0 = [10.0, 2.0]

    # setup dyn prior
    M = np.zeros([ld] * 2)
    K = 0.5 * np.eye(ld)
    n0 = 10
    S0 = 0.1 * np.eye(ld)

    # remaining priors
    P0 = 1e-6 * np.eye(ld)
    r0 = 3
    R0 = 1e-6 * np.eye(od)

    # create priors
    hdp_prior = StickyHDPPrior(max_z, alpha_kappa0, rho0, gamma0, 'uniform')
    mniw_prior = MNIWPrior(M, K, n0, S0)
    latent_prior = P0
    emission_prior = [r0, R0]

    # final model
    folder = join(root, name)
    model = StickyHDPSLDSHMMModel(ld, od, hdp_prior, mniw_prior, latent_prior, emission_prior, C)
    generate_hdp_slds_groundtrouth(folder, num_traces, num_steps, model)


def generate_ard_data(root='./data_packs', name='hdp_slds_ard'):

    print_line()
    print_line("Generate HDP-SLDS-HMM using ARD prior")

    # some general settings
    num_traces = 50
    num_steps = 100
    ld = 3
    od = 2
    max_z = 5

    # projection
    C = np.hstack((np.eye(od), np.zeros([od, ld-od])))

    # setup hdp prior
    alpha_kappa0 = [15.0, 3.0]
    rho0 = [5.0, 2.0]
    gamma0 = [10.0, 2.0]

    # setup dyn prior
    alpha0 = [16.0, 8.0]
    n0 = 5
    S0 = 2 * np.eye(ld)

    # remaining priors
    P0 = 1e-6 * np.eye(ld)
    r0 = 3
    R0 = 1e-6 * np.eye(od)

    # create priors
    hdp_prior = StickyHDPPrior(max_z, alpha_kappa0, rho0, gamma0, 'uniform')
    ard_prior = ARDPrior(alpha0, n0, S0)
    latent_prior = P0
    emission_prior = [r0, R0]

    # final model
    folder = join(root, name)
    model = StickyHDPSLDSHMMModel(ld, od, hdp_prior, ard_prior, latent_prior, emission_prior, C)
    generate_hdp_slds_groundtrouth(folder, num_traces, num_steps, model)


generate_ard_data()
generate_mniw_data()
