import numpy as np

from slds.densities import gaussian_logpdf


class StickyHDPSLDSHMMModelBlockZMixin:
    def slds_post_block_sample_z(self) -> None:
        """Block-sample the mode path given the latent trajectory.

        Messages and the categorical draws are computed in log space. Time 0
        has no dynamics likelihood, matching the stored-array convention.
        """

        log_lik = self._log_transition_likelihood()
        log_messages = self._log_backward_messages(log_lik)
        self._sample_modes_from_log_messages(log_lik, log_messages)

    def _log_transition_likelihood(self) -> np.ndarray:
        """Log transition density for every trace, time, and mode.

        Returns:
            Array of shape ``(traces, steps, modes)``. Time 0 is zero because
            that site has no stored parent.
        """

        trace_count, step_count, _ = self.x.shape
        mode_count = self.dynamics.A.shape[0]
        log_lik = np.zeros((trace_count, step_count, mode_count))
        if step_count < 2:
            return log_lik
        parents = self.x[:, :-1]
        children = self.x[:, 1:]
        for mode in range(mode_count):
            mean = parents @ self.dynamics.A[mode].T
            log_lik[:, 1:, mode] = gaussian_logpdf(children, mean, self.dynamics.Sigma[mode])
        return log_lik

    def _log_backward_messages(self, log_lik: np.ndarray) -> np.ndarray:
        """Backward mode messages in log space.

        ``log m_t(k) = logsumexp_j(log pi_{kj} + log lik(x_{t+1}|x_t, j) + log m_{t+1}(j))``.
        The current likelihood is applied at the sampling step. The terminal
        message is one.

        Args:
            log_lik: Transition log densities, shape ``(traces, steps, modes)``.

        Returns:
            Normalized log messages of the same shape.
        """

        _traces, step_count, _modes = log_lik.shape
        with np.errstate(divide="ignore"):
            log_pi = np.log(np.asarray(self.hdp.pi, dtype=np.float64))
        log_messages = np.zeros_like(log_lik)
        for time in reversed(range(step_count - 1)):
            terms = (
                log_pi[None, :, :]
                + log_lik[:, time + 1, None, :]
                + log_messages[:, time + 1, None, :]
            )
            log_messages[:, time, :] = np.logaddexp.reduce(terms, axis=2)
            log_messages[:, time, :] -= np.logaddexp.reduce(
                log_messages[:, time, :], axis=1, keepdims=True
            )
        return log_messages

    def _sample_modes_from_log_messages(
        self, log_lik: np.ndarray, log_messages: np.ndarray
    ) -> None:
        """Draw each mode from the normalized log categorical.

        Args:
            log_lik: Transition log densities.
            log_messages: Normalized backward log messages.
        """

        trace_count, step_count, mode_count = log_lik.shape
        pi = np.asarray(self.hdp.pi, dtype=np.float64)
        beta = np.asarray(self.hdp.beta, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_pi = np.log(pi)
            log_beta = np.log(beta)
        for time in range(step_count):
            if time > 0:
                log_score = log_pi[self.z[:, time - 1]] + log_lik[:, time] + log_messages[:, time]
            else:
                log_score = log_beta + log_lik[:, time] + log_messages[:, time]
            for trace in range(trace_count):
                row = log_score[trace]
                finite = np.isfinite(row)
                if not np.any(finite):
                    probs = np.full(mode_count, 1.0 / mode_count)
                else:
                    shifted = np.exp(np.where(finite, row, -np.inf) - np.max(row[finite]))
                    total = np.sum(shifted)
                    probs = shifted / total
                self.z[trace, time] = np.random.choice(np.arange(mode_count), p=probs)
