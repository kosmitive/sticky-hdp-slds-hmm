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
from scipy.stats import invwishart, matrix_normal


class MNIWPrior:

    fn = 'mniw_prior.json'
    fn_M = 'M.npy'
    fn_K = 'K.npy'
    fn_S0 = 'S0.npy'

    def __init__(self, M, K, n0, S0):
        """This initializes a new mniw prior, which can be e.g. used to set a prior
        on a transformation and noise covariance.

        :param M: The mean of the matrix normal.
        :param K: The column variance
        :param n0: The dof of the inverse wishart.
        :param S0: The scale of the inverse wishart.
        """

        self.M = M
        self.K = K
        self.n0 = n0
        self.S0 = S0

    def prior_sample(self, num):

        # first of all sample the covariance
        Sigma = invwishart.rvs(self.n0, self.S0, num)

        # get dims and reserve space
        nd = self.S0.shape[0]
        A = np.empty([num, nd, nd])

        # fill at separately
        for z in range(num):
            A[z] = matrix_normal.rvs(self.M, Sigma[z], self.K)

        # pass back the sample
        return [A, Sigma]

    @staticmethod
    def load(folder):
        """Load the prior from the specified folder."""

        d = load_json(folder, MNIWPrior.fn)
        n0 = d['n0']
        M = np.load(join(folder, MNIWPrior.fn_M))
        K = np.load(join(folder, MNIWPrior.fn_K))
        S0 = np.load(join(folder, MNIWPrior.fn_S0))
        return MNIWPrior(M, K, n0, S0)

    def store(self, folder):
        """Store the prior in the specified folder."""

        # store the dict
        sdict = { 'n0': self.n0 }
        save_json(folder, MNIWPrior.fn, sdict)
        np.save(join(folder, MNIWPrior.fn_M), self.M)
        np.save(join(folder, MNIWPrior.fn_K), self.K)
        np.save(join(folder, MNIWPrior.fn_S0), self.S0)