import numpy as np
from misc.disc_sequences import count_state_transitions
from scipy.stats import bernoulli, beta


class StickyHDPSLDSHMMModelHDPMixin:
    def calc_transitions(self):

        self.transitions = count_state_transitions(self.z, self.hdp_prior.max_z)

    def sample_post_auxiliary_vars(self) -> None:
        """Sample table counts, sticky overrides, and the informative counts.

        Overrides are stored on ``override_counts`` so they do not replace
        the emission noise in ``w``. ``m_hat`` is ``m`` with those overrides
        removed from the diagonal, as in Fox et al. Eq. (B.20).
        """

        beta = self.hdp.beta
        alpha_kappa = self.hdp.alpha_plus_kappa
        rho = self.hdp.rho

        # calc
        kappa = rho * alpha_kappa
        alpha = (1 - rho) * alpha_kappa

        transitions = self.transitions

        trunc_level = np.size(beta)
        m = np.zeros([trunc_level] * 2)

        # Fox et al., sticky HDP-HMM supplement, Algorithm 1 step 3(a):
        # the n-th customer (n starting at 0) opens a table with probability
        # (alpha * beta_k + kappa * 1_{j=k}) / (n + alpha * beta_k + kappa * 1_{j=k}).
        # The first customer therefore always opens a table.
        for j in range(trunc_level):
            for k in range(trunc_level):
                count = int(transitions[j, k])
                if count == 0:
                    continue
                term = alpha * beta[k] + (kappa if j == k else 0.0)
                seats = np.arange(count)
                probs = term / (seats + term)
                m[j, k] = np.random.binomial(1, probs).sum()

        # Override indicators on self-transition tables only.
        # Stored separately from the emission noise ``self.w``.
        rho = kappa / (alpha + kappa)
        p = rho / (rho + beta * (1 - rho))
        override_counts = np.asarray(
            [np.random.binomial(m[j, j], p[j]) for j in range(trunc_level)],
            dtype=np.float64,
        )
        m_hat = m - np.diag(override_counts)

        self.m = m
        self.m_hat = m_hat
        self.override_counts = override_counts

    def sample_post_hdp_rho(self):

        m = self.m
        w = self.override_counts
        prior = self.hdp_prior.rho0

        sw = np.sum(w)
        a = sw + np.asarray(prior[0])
        b = np.sum(m) - sw + prior[1]

        # sample rho
        rho = np.random.beta(a, b)
        self.hist_rho.append((a, b))
        self.hdp.rho = rho

    def sample_post_hdp_gamma(self):

        m_hat = self.m_hat
        prior = self.hdp_prior.gamma0
        gamma = self.hdp.gamma

        # first of all sample auxiliary vars
        bigk = np.count_nonzero(np.greater(np.sum(m_hat, axis=0), 0))

        # and another one
        summed_m_hat = np.sum(m_hat)
        eta = np.random.beta(gamma + 1, summed_m_hat + 1e-6)

        # sample gamma
        fp = prior[0] + bigk
        sp = prior[1] - np.log(eta)
        p_mixing = summed_m_hat / (summed_m_hat + gamma)
        mixing = np.random.binomial(n=1, p=p_mixing)

        # now sample gamma
        a = fp - mixing
        b = 1 / sp
        gamma = np.random.gamma(a, b)
        self.hist_gamma.append((a, b))
        self.hdp.gamma = gamma

    def sample_post_hdp_alpha_kappa(self):

        transitions = self.transitions
        prior = self.hdp_prior.alpha_kappa0
        alpha_kappa = self.hdp.alpha_plus_kappa
        n_rows = np.sum(transitions, axis=1)

        # Auxiliary variables exist only for restaurants that served customers.
        # Empty truncation states contribute a factor of 1 and are skipped.
        active = np.flatnonzero(n_rows > 0)
        r = np.asarray([beta.rvs(alpha_kappa + 1, n_rows[j]) for j in active])
        s = np.asarray([bernoulli.rvs(n_rows[j] / (n_rows[j] + alpha_kappa)) for j in active])

        # sample the sum of alpha and kappa
        summed_s = np.sum(s) if s.size else 0.0
        summed_r = np.sum(np.log(r)) if r.size else 0.0

        # get a and b
        a = prior[0] + np.sum(self.m) - summed_s
        b = 1 / (prior[1] - summed_r)
        alpha_kappa = np.random.gamma(a, b)

        self.hist_alpha_kappa.append((a, b))
        self.hdp.alpha_plus_kappa = alpha_kappa

    def sample_post_hdp_pi(self):

        transitions = self.transitions
        beta = self.hdp.beta
        alpha_kappa = self.hdp.alpha_plus_kappa
        rho = self.hdp.rho

        # calc
        kappa = rho * alpha_kappa
        alpha = (1 - rho) * alpha_kappa

        trunc_level = transitions.shape[0]
        pi = np.empty(2 * [trunc_level], np.float64)

        # update pi
        id = np.eye(trunc_level)
        for k in range(trunc_level):
            probs = alpha * beta + transitions[k] + id[k] * kappa
            pi[k] = np.random.dirichlet(probs)

        self.hdp.pi = pi

    def sample_post_hdp_beta(self):

        trunc_level = self.hdp_prior.max_z
        summed_m_hat = np.sum(self.m_hat, axis=0)
        self.hdp.beta = np.random.dirichlet(self.hdp.gamma / trunc_level + summed_m_hat)
