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


import scipy.special
from scipy.stats import invwishart

from slds.slds_seq_sample_z import *


class StickyHDPSLDSHMMModelEmissionMixin:

    def slds_post_sample_r(self):

        num_traces, num_steps, _ = self.y.shape
        dof = num_traces + num_steps + self.emission_prior[0]
        scale = self.emission_prior[1]

        for d in range(num_traces):
            for t in range(num_steps):
                vec = self.y[d, t] - self.C @ self.x[d, t]
                new_t = np.outer(vec, vec)
                scale += new_t

        # now sample r
        self.R = invwishart.rvs(dof, scale)
