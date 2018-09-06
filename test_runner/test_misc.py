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

from misc.disc_dist_divergences import minimum_hamming_distance
from misc.disc_sample import sample_cat


def test_hamming_dist():

    for i in range(100):

        # create some distribution
        num_modes = 10
        dist = np.random.randint(0, 1000, num_modes)

        # now sample a sequence from the distribution
        num_elements = 1000
        samples = sample_cat(dist, num_elements)
        o_samples = np.copy(samples)

        # now create random numer
        h_dist = np.random.randint(0, 100)
        p = np.random.permutation(1000)

        # change these elements
        for k in range(h_dist):
            r = p[k]
            s = samples[r]
            old_dist = dist[s]
            dist[s] = 0
            samples[r] = sample_cat(dist)
            dist[s] = old_dist

        res, _ = minimum_hamming_distance(samples, o_samples)
        assert res == h_dist


def test_misc():
    test_hamming_dist()

test_misc()