import numpy as np


def regularize_matrix(mat, lam=10e-5):
    cs1, cs2 = mat.shape
    assert cs1 == cs2
    return mat + lam * np.eye(cs1)


def hom_mat(theta, t):
    return np.array(
        [[np.cos(theta), -np.sin(theta), t[0]], [np.sin(theta), np.cos(theta), t[1]], [0, 0, 1]]
    )


def pad(x, val=1.0, left=1, t=1, r=1, b=1):
    h, w = x.shape
    x_new = np.ones([h + t + b, w + left + r], np.float64) * val
    x_new[t : t + h + 1, left : left + w + 1] = x
    return x_new
