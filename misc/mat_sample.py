import numpy as np


def mat_sample_uniform_transition(tr_dims, num=1):
    """This method samples some uniform transition matrices, in the
    sense that their columns are normalized to one.

    :param num_matrices: The number of matrices to create.
    :param tr_dims: The number of dimensions the transition matrix should have.
    :return: The updated matrix element.
    """

    mats = np.random.uniform(0, 1, [num, tr_dims, tr_dims])
    div = np.expand_dims(np.sum(mats, 1), 1)

    return mats / div
