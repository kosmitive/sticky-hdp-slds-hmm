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


from misc.hdd_save_load import save_json, load_json
import numpy as np

from misc.stick_breaking import stick_breaking_construction_sticky_hdp


class StickyHDPPrior:

    fn = 'sticky_hdp_prior.json'

    def __init__(self, max_z, alpha_kappa0, rho0, gamma0, H0):
        """This initializes a HDP Prior with all relevant information."""

        # 0save paramse
        self.alpha_kappa0 = alpha_kappa0
        self.rho0 = rho0
        self.gamma0 = gamma0
        self.H0 = H0
        self.max_z = max_z

    def prior_sample(self):

        # sample some value for each of them
        alpha_kappa = np.random.gamma(self.alpha_kappa0[0], 1 / self.alpha_kappa0[1])
        rho = np.random.beta(self.rho0[0], self.rho0[1])
        gamma = np.random.gamma(self.gamma0[0], 1 / self.gamma0[1])

        # calc alpha and kappa
        kappa = rho * alpha_kappa
        alpha = (1 - rho) * alpha_kappa

        # create beta and pi
        beta, pi = stick_breaking_construction_sticky_hdp(alpha, kappa, gamma, self.max_z, self.H0)

        # pass back
        return [alpha_kappa, rho, gamma, beta, pi]

    def max_likely_sample(self):

        alpha_kappa = self.alpha_kappa0[0] / self.alpha_kappa0[1]
        rho = self.rho0[0] / (self.rho0[0] + self.rho0[1])
        gamma = self.gamma0[0] / self.gamma0[1]

        # calc alpha and kappa
        kappa = rho * alpha_kappa
        alpha = (1 - rho) * alpha_kappa

        # create beta and pi
        beta, pi = stick_breaking_construction_sticky_hdp(alpha, kappa, gamma, self.max_z, self.H0)

        # pass back
        return [alpha_kappa, rho, gamma, beta, pi]

    @staticmethod
    def load(folder):
        """Load the prior from the specified folder."""

        d = load_json(folder, StickyHDPPrior.fn)
        return StickyHDPPrior(d['max_z'],
                              [d['alpha_kappa_shape'], d['alpha_kappa_rate']],
                              [d['rho_shape_a'], d['rho_shape_b']],
                              [d['gamma_shape'], d['gamma_rate']],
                              d['H0'])

    def store(self, folder):
        """Store the prior in the specified folder."""

        # store the dict
        sdict = {
            'alpha_kappa_shape': self.alpha_kappa0[0],
            'alpha_kappa_rate': self.alpha_kappa0[1],
            'rho_shape_a': self.rho0[0],
            'rho_shape_b': self.rho0[1],
            'gamma_shape': self.gamma0[0],
            'gamma_rate': self.gamma0[1],
            'H0': self.H0,
            'max_z': self.max_z
        }

        save_json(folder, StickyHDPPrior.fn, sdict)
