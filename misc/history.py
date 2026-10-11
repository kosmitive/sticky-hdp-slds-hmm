import os

import numpy as np


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
