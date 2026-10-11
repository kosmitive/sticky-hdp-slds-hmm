import math as m

import numpy as np


def sv_gamma_pdf(x, shape, rate):
    norm = rate**shape / m.gamma(shape)
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
