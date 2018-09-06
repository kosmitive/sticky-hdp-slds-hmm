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

from misc.hdd_save_load import save_json, load_json
import numpy as np
from scipy.stats import gamma, invwishart, multivariate_normal


class ARDPrior:

    # define function names
    fn = 'ard_prior.json'
    fn_S0 = 'S0.npy'

    def __init__(self, alpha0, n0, S0):

        # save params
        self.alpha0 = alpha0
        self.n0 = n0
        self.S0 = S0

    def prior_sample(self, num):

        # first of all sample the covariance
        Sigma = invwishart.rvs(self.n0, self.S0, num)

        # perform a simple forward sample
        nd = self.S0.shape[0]
        alpha = gamma.rvs(self.alpha0[0], 1 / self.alpha0[1], size=(num, nd))

        # use the generated alpha to generate the modes
        cov = np.diag(1 / alpha.repeat(nd))
        A = multivariate_normal.rvs(np.zeros(num * nd ** 2), cov)

        # bring in correct form
        A = A.reshape([num, nd, nd]).transpose([0, 2, 1])

        # pass back the sample
        return A, Sigma, alpha

    @staticmethod
    def load(folder):
        """Load the prior from the specified folder."""

        d = load_json(folder, ARDPrior.fn)
        alpha_shape = d['alpha_shape']
        alpha_rate = d['alpha_rate']
        n0 = d['n0']
        S0 = np.load(join(folder, ARDPrior.fn_S0))

        return ARDPrior([alpha_shape, alpha_rate], n0, S0)

    def store(self, folder):
        """Store the prior in the specified folder."""

        # store the dict
        sdict = {
            'n0': self.n0,
            'alpha_rate': self.alpha0[1],
            'alpha_shape': self.alpha0[0],
        }

        save_json(folder, ARDPrior.fn, sdict)
        np.save(join(folder, ARDPrior.fn_S0), self.S0)
