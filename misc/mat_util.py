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


def regularize_matrix(mat, lam=10e-5):
    cs1, cs2 = mat.shape
    assert cs1 == cs2
    return mat + lam * np.eye(cs1)


def hom_mat(theta, t):
    return np.array([[np.cos(theta), -np.sin(theta), t[0]],
                     [np.sin(theta), np.cos(theta), t[1]],
                     [0, 0, 1]])


def pad(x, val=1.0, l=1, t=1, r=1, b=1):
    h, w = x.shape
    x_new = np.ones([h+t+b,w+l+r], np.float64) * val
    x_new[t:t+h+1,l:l+w+1] = x
    return x_new
