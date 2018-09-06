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
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('text', usetex=True)

from mpl_toolkits.mplot3d import Axes3D


def plot_trajectories(root, name, trajectories, title, ylabel, style='normal'):
    """This method can plot trajectories

    :param root: The root folder for image.
    :param name: The name of the file.
    :param trajectories: The trajectories
    :param title: The title of the plot.
    :param ylabel: The label for y axis
    """

    print("Plotting trajectories to {}".format(name), end='')

    # there are two modes depending on the number of dimensions
    # of the trajectories
    one_only = False
    if np.ndim(trajectories) == 2:
        one_only = True
        trajectories = np.expand_dims(trajectories, 0)

    # remap sizes
    trajectories = trajectories if one_only else trajectories.transpose([0, 2, 1])
    num_traj, num_trace, num_step = trajectories.shape
    fig = plt.figure()

    # create filename and start plot
    for i in range(num_traj):
        print('.', end='')
        # get file name
        rn = "{}.pdf".format(name) if one_only else "{}_{}.pdf".format(name, i)
        filename = os.path.join(root, rn)
        ax = __plot_setup(1)
        ax = ax[0]

        # transpose if necessary
        if style is 'normal':
            [plt.plot(trajectories[i,j]) for j in range(num_trace)]
            plt.ylabel(ylabel)
            plt.xlabel("Step $t$")
            plt.xlim([0, num_step-1])

            # only integer values
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        elif style is 'draw':
            if trajectories.shape[1] == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(trajectories[i,0], trajectories[i,1], trajectories[i,2])

            elif trajectories.shape[1] == 2:
                ax.plot(trajectories[i,0], trajectories[i,1])

        # set labels and finalize
        plt.title(title.format(i))
        __plot_finish_off(filename)

    print('.')


def plot_discrete_distribution(root, name, dist, title, labels, digits=3):
    """This method can be used to generate and save the plot
    of a discrete distribution.

    :param root: The root folder for image.
    :param name: The name of the file.
    :param dist: The distribution as a vector [K] where K is the number of elements.
    :param title: The title for the plot
    :param labels: If 1 dimensional 1 element and if 2 dimensional a list of 2 labels
    """

    print("Plotting discrete distribution to {}".format(name))

    # get number of elements
    filename = os.path.join(root, "{}.pdf".format(name))

    # make a list out of the parameter
    if not isinstance(dist, list):
        dist = [dist]

    # make a list out of the parameter
    if not isinstance(title, list):
        title = [title]

    # number of plots
    num_plots = len(dist)

    # iterate over all plots
    fig = plt.figure()
    ax = __plot_setup(num_plots)

    # iterate over all plots single
    for i in range(num_plots):

        # expand if neccessary
        is_one_dimensional = np.ndim(dist[i]) == 1
        nel = np.expand_dims(dist[i], 1) if is_one_dimensional else dist[i]
        c_vals = ax[i].imshow(nel, interpolation='nearest')

        # if it is one-dimensional
        if is_one_dimensional:
            num_elements_h, _ = nel.shape
            ax[i].set_yticks(np.arange(num_elements_h))
            ax[i].set_ylabel(labels)
            ax[i].set_xticks([])

        else:
            num_elements_h, num_elements_h = nel.shape
            ax[i].set_ylabel(labels[0])
            ax[i].set_xlabel(labels[1])
            ax[i].set_xticks(np.arange(num_elements_h))
            ax[i].set_yticks(np.arange(num_elements_h))

        ax[i].set_title(title[i])
        __plot_text_on_cells(ax[i], nel, digits)

    plt.colorbar(c_vals)
    __plot_finish_off(filename)


def __plot_setup(num_plots):

    # generate figure
    plt.clf()
    if num_plots > 1:
        fig, ax = plt.subplots(1, num_plots, sharey=True)

    else:
        ax = [plt.axes()]

    return ax


def __plot_text_on_cells(ax, tdist, digits):

    h, w = tdist.shape
    mid = (np.max(tdist) - np.min(tdist)) * 0.5
    for i in range(h):
        for j in range(w):
            c = "black" if tdist[i, j] > mid else "w"
            ax.text(j, i, ("{:0." + str(digits) + "f}").format(tdist[i, j]),
                    ha="center", va="center", color=c)


def __plot_finish_off(filename):

    # set limits
    plt.tight_layout()
    # save the current plot
    plt.savefig(filename)

