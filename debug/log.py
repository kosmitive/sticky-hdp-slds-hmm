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
import os
import time
from misc.output import print_mat_statistics


class TestLogger:

    """Simple class which represents a log token."""
    def __init__(self, f_path):
        self.f_path = f_path
        self.num_cats = 0

        # create dir if necessary
        if not os.path.exists(f_path):
            os.mkdir(f_path)

    def create_category(self, testname, category):
        """This method creates a new test_runner using the given testname."""

        # get path to new category.

        self.init_test(testname)
        new_cat = os.path.join(self.f_path, testname, category)

        if not os.path.exists(new_cat):
            os.mkdir(new_cat)

        return new_cat

    def init_test(self, testname):
        """This method creates a new category inside of the folder."""

        # get path to new category.

        new_test = os.path.join(self.f_path, testname)

        # create dir if neceyss
        if not os.path.exists(new_test):
            os.mkdir(new_test)

        return new_test

    def log_matrix(self, test, id, matrix):
        """This method logs the passed matrix, it

        :param id: The id of the matrix. Remember repeating calls must be called with the same matrix.
        :param matrix: The matrix to log itself.
        """

        # print_mat_statistics(id, matrix)
        cat = self.create_category(test, "matrices")
        mat_name = os.path.join(cat, id)
        d, _ = matrix.shape

        # now simply save some files
        eig_file = os.path.join(mat_name, "eig.npy")
        mat_file = os.path.join(mat_name, "mat.npy")
        eig_basis_file = os.path.join(mat_name, "eig_basis.npy")

        # if the path doesn't exist
        if not os.path.exists(mat_name):

            # create it and fill initially with some garbage
            os.mkdir(mat_name)

            # define the old values
            old_eig = np.empty([0, d], dtype=np.complex64)
            old_mat = np.empty([0, d, d], dtype=np.float64)
            old_eig_basis = np.empty([0, d, d], dtype=np.complex64)

        else:

            # load the text in the folder
            old_eig = np.load(eig_file)
            old_mat = np.load(mat_file)
            old_eig_basis = np.load(eig_basis_file)

        # calc eigenvalues
        evals, evecs = np.linalg.eig(matrix)
        evals = np.sort(evals)

        X = np.vstack((old_eig, np.expand_dims(evals, 0)))
        np.save(eig_file, X)

        X = np.vstack((old_mat, np.expand_dims(matrix, 0)))
        np.save(mat_file, X)

        X = np.vstack((old_eig_basis, np.expand_dims(evecs, 0)))
        np.save(eig_basis_file, X)


def create_logtoken(root):
    """Specify a root directory and receive a token stating which
    directory the operation will happen to

    :param root: The root, like a collection directory.
    :return: The file path itself should be returned.
    """

    # load the traces of observations
    folder = time.strftime("%Y%m%d-%H%M%S")
    f_path = os.path.join(root, folder)

    return TestLogger(root)