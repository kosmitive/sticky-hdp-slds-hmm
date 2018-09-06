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

from misc.disc_dist_divergences import regularize_matrix


def sv_gamma_pdf(x, shape, rate):
    norm = rate ** shape / m.gamma(shape)
    un_norm = x ** (shape - 1) * np.exp(-rate * x)
    return norm * un_norm


def sv_gauss_pdf(x, mu, sigma):

    n = len(mu)
    centered_x = x - mu
    m_dist = -0.5 * np.inner(centered_x, np.linalg.inv(sigma) @ centered_x)
    norm_c = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
    return norm_c * np.exp(m_dist)


def sv_beta_pdf(x, a, b):
    """Calculate pdf of beta(x|a,b).

    :param x: Vector [D] of all elements.
    :param a: The shape parameter a
    :param b: The shape parameter b
    :return: Vector [D] containing the pdfs.
    """

    # get norm constant
    beta_norm = (m.gamma(a) * m.gamma(b)) / m.gamma(a + b)

    # calc un-normalized and return normalized
    un_norm = x ** (a - 1) * (1 - x) ** (b - 1)
    return un_norm / beta_norm