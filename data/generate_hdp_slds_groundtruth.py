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
import os
import shutil

from os.path import join

from model.ard_prior import ARDPrior
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel
from model.mniw_prior import MNIWPrior
from misc.output import print_line
from misc.plotting import plot_discrete_distribution, plot_trajectories
from model.sticky_hdp_prior import StickyHDPPrior


def generate_hdp_slds_groundtrouth(folder, num_traces, num_steps, model):
    """This method generates a groundtrouth package and saves it. The parameters have to
    be adjusted below in order to get different outcomes.

    :param root: This represents the root folder of the data packages, which should be saved.
    :param name: The name of the data package itself.
    """

    print_line()
    print("Empty/Create folder {}".format(folder))

    # remove folder if exists
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    # simulate the slds traces
    print("Generating data...")
    model.simulate(num_traces, num_steps)

    # store the model
    print("Storing numerical data...")
    model.store(folder)

    # now generate an image folder
    img_path = os.path.join(folder, 'img')
    os.mkdir(img_path)

    # generate path for beta plot
    print("Creating discrete plots...")
    plot_discrete_distribution(img_path, "beta", model.hdp_params[3], r'$\beta[z]$ --- Parent Dist.', 'z')
    plot_discrete_distribution(img_path, "pi", model.hdp_params[4], r'$\pi[z_t, z_{t+1}]$ --- Transition Probs.',
                               ['$z_t$', '$z_{t+1}$'])

    # plot the modes and the covariances as well
    ld = model.ld
    mid = [model.slds_params[0].reshape([-1,  model.slds_params[0].shape[-1]]), model.slds_params[1].reshape([model.hdp_prior.max_z * ld,  ld])]
    plot_discrete_distribution(img_path, "dynamics", mid, [r'$A$ --- Mode Projection.', r'$\Sigma$ --- Mode Covariance.'], ['$x_{t+1}$', '$x_t$'], digits=2)

    # plot the trajectories
    print("Creating trajectories plots...")
    num_trajs = 5
    plot_trajectories(img_path, "modes", model.z[:num_trajs], "$z_t$ --- Mode Sequence", "Mode")
    plot_trajectories(img_path, "latent", model.x[:num_trajs], "$x_t$ --- Latent Sequence {}", "Vals")
    plot_trajectories(img_path, "latent_3d", model.x[:num_trajs], "$x_t$ --- Latent Sequence {}", "Vals", style='draw')
    plot_trajectories(img_path, "emission", model.y[:num_trajs], "$y_t$ --- Observable Sequence {}", "Vals")
    plot_trajectories(img_path, "emission_3d", model.y[:num_trajs], "$y_t$ --- Observable Sequence {}", "Vals", style='draw')
    plot_trajectories(img_path, "latent_noise", model.e[:num_trajs], "$e_t$ --- Latent Noise {}", "Vals")
    plot_trajectories(img_path, "emission_noise", model.w[:num_trajs], "$w_t$ --- Emission Noise {}", "Vals")

    print_line()