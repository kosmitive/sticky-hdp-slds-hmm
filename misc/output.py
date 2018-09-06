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


import matplotlib.pyplot as plt
import numpy as np

from misc.ansi_color_codes import ACC


def gen_plot(timeline, filename, title):
    if not isinstance(timeline, list):
        timeline = [timeline]

    plt.figure(10000)
    plt.clf()
    for i in range(len(timeline)): plt.plot(timeline[i])
    plt.title(title)
    plt.savefig(filename)


def print_dbl_line(num=150):
    print("=" * num)


def print_trace(trace_num, mode_seq, po, observations, sc_mode_seq, sc_po, sc_observations, max_steps=20):

    _, num_steps, _ = observations.shape
    print_line()
    for s in range(np.minimum(max_steps, num_steps)):
        print("\t| ", mode_seq[trace_num, s], " ", ACC.OkBlue, "[", sc_mode_seq[trace_num, s], "] \t", ACC.End, end='')
        print("-> [", *print_array(po[trace_num, s]), "] ", ACC.OkBlue, "[", *print_array(sc_po[trace_num, s]), "]", ACC.End, end='')
        print("-> [", *print_array(observations[trace_num, s]), "] ")

    [print("\t" * 2 + "  ." + "\t" * 9 + "." + "\t" * 10 + "  ." + "\t" * 9 + "  .") for _ in range(3)]


def print_array(a, format_string ='{0:+.8f}'):
    return [format_string.format(v,i) for i,v in enumerate(a)]


def print_line(text=None, num=150):
    print("-" * num)
    if text is not None:
        print("--- ", end='')
        print(text, end='')
        print(" ", "-" * (num - 6 - len(text)))


def print_mat_statistics(id, a):

    # extract, eigenvalues and eigenvectors
    np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    evals, evecs = np.linalg.eig(a)

    # sort the values
    sorting = evals.argsort()[::-1]
    evals = evals[sorting]
    evecs = evecs[:, sorting]

    # calc condition number
    cond_num = np.abs(evals[0]) / np.abs(evals[-1])

    # do the printing
    print_line()
    print("Matrix [{}]".format(id))
    print_line("Properties")
    print("\t| COND_NUM = {:0.6f}".format(cond_num))
    print("\t| EIG = \t", end='')
    for i in range(len(evals)):
        print(evals[i])
        if i < len(evals) - 1:
            print("\t\t\t\t", end='')

    # print matrix
    print_line("Matrix")
    [print(a[i]) for i in range(len(a))]

    # print eigen vector basis
    print_line("U")
    [print("u{} -> {} -> {}".format(i, evals[i], evecs[:,i])) for i in range(len(a))]
