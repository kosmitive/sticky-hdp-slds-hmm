import numpy as np


def count_state_transitions(sequences, max_state):

    # split dims of shape
    num_traces, num_steps = sequences.shape

    # create count space
    counts = np.zeros(2 * [max_state])

    # iterate over all dimensions
    for d in range(num_traces):
        # get current state estimate
        z_t = sequences[d, 0]
        for t in range(1, num_steps):
            # map states
            z_pt = int(z_t)
            z_t = int(sequences[d, t])

            # increase count
            counts[z_pt, z_t] += 1

    return counts
