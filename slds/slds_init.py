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
from misc.disc_sample import sample_cat


class StickyHDPSLDSHMMModelInitMixin:

    def slds_init_z(self):

        num_traces, num_steps, _ = self.y.shape
        beta = self.hdp_params[3]
        pi = self.hdp_params[4]

        # now we sample the mode sequence
        max_state = len(beta)
        mode_seq = np.empty([num_traces, num_steps], np.int64)
        mode_seq[:, 0] = np.random.choice(max_state, p=beta, size=[num_traces])

        # iterate over
        for trace in range(num_traces):
            for step in range(num_steps - 1):
                z = mode_seq[trace, step]
                mode_seq[trace, step + 1] = np.random.choice(max_state, p=pi[z])

        self.z = mode_seq
