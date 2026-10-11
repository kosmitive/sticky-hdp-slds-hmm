import numpy as np


class StickyHDPSLDSHMMModelInitMixin:
    def slds_init_z(self):

        num_traces, num_steps, _ = self.y.shape
        beta = self.hdp.beta
        pi = self.hdp.pi

        # now we sample the mode sequence
        max_state = len(beta)
        mode_seq = np.empty([num_traces, num_steps], np.int64)
        mode_seq[:, 0] = np.random.choice(max_state, p=beta, size=[num_traces])

        # iterate over
        for trace in range(num_traces):
            for step in range(num_steps - 1):
                z = mode_seq[trace, step]
                mode_seq[trace, step + 1] = np.random.choice(max_state, p=pi[z])

        self.z = mode_seq
