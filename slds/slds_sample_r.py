import numpy as np
from scipy.stats import invwishart


class StickyHDPSLDSHMMModelEmissionMixin:
    def slds_post_sample_r(self) -> None:
        """Sample the shared emission covariance from its inverse-Wishart posterior.

        Fox et al. (arXiv:1003.3829), Eq. (34), give
        ``R ~ IW(r0 + T, R0 + S_R)`` where ``T`` is the number of
        observations. Multiple traces contribute ``num_traces * num_steps``
        observations, not their sum.
        """

        num_traces, num_steps, _ = self.y.shape
        dof = num_traces * num_steps + self.emission_prior[0]
        scale = np.array(self.emission_prior[1], dtype=np.float64, copy=True)

        for d in range(num_traces):
            for t in range(num_steps):
                vec = self.y[d, t] - self.C @ self.x[d, t]
                new_t = np.outer(vec, vec)
                scale += new_t

        # now sample r
        self.R = invwishart.rvs(dof, scale)
