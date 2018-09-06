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


def mv_gauss_logpdf(x, mu, sigma):
    """This method is able to calculate the logits of a multivariate gaussian. Uses
    two calculation tricks, to speed up the evaluation. The output is a floating number

        p = log N(x|mu,sigma)

    :param x: The value for which the probability value should be estimated.
    :param mu: The mean of the multivariate gaussian distribution.
    :param sigma: The covariance matrix of the multivariate gaussian distribution.
    :return: The logits of a gaussian multivariaten distribution.
    """

    # check if compatible
    if x.ndim == 1:
        x = np.expand_dims(x, 0)

    num_examples, num_dim_x = x.shape
    if isinstance(mu, int):
        emu = np.ones([1, num_dim_x]) * mu
    elif mu.ndim == 1 and len(mu) == num_dim_x:
        emu = np.expand_dims(mu, 0)
    elif mu.ndim == 2 and mu.shape == x.shape:
        emu = mu
    else: assert False

    # C is pos definite and calc cholesky
    scaled_cov = 2 * np.pi * sigma
    lower_triang = np.linalg.cholesky(scaled_cov)
    log_scaled_cov = np.sum(np.log(np.diag(lower_triang)))

    # now calc exponential part
    c_sigma = np.linalg.cholesky(sigma)
    centered_x = (x - emu).transpose()
    coeff = np.linalg.solve(c_sigma, centered_x)
    mah_dist = np.dot(coeff.transpose(), coeff)

    return -(log_scaled_cov + 0.5 * mah_dist)