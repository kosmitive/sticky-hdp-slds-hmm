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


class History:
    """This class represents a history module."""

    def __init__(self, steps):
        self.mem = {}
        self.T = steps

    def add_cat(self, name, dim=1):

        # if already in throw error
        if name in self.mem:
            raise ValueError("The name already exists.")

        # add to memory
        self.mem[name] = np.empty([self.T, dim])


    def add_value(self, cat, step, value):

        if cat not in self.mem:
            raise ValueError("You have to create a cat first")

        self.mem[cat][step] = value

    def save(self, res):

        # save all fields
        for v in self.mem:
            np.save(os.path.join(res, v), self.mem[v])