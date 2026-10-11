import numpy as np


class StickyHDPSLDSHMMModelKalmanYMixin:
    def slds_forward_kalman_filter_gains(self):
        """Use numerically stable forward kalman filter from paper of fox. Provided in the
        report of 1003.3829

        :param conf: The configuration, priors ...
        :param mode_seq: The truncated mode sequence [DxT] of maximum L used modes.
        :param observations: The observation matrix [DxTxO] where each column is one time step.
        :return: The kalman matrices Lambda [Dx(T+1)xHxH] and theta [Dx(T+1)xH] computed throughout.
        """

        # rename a little bit
        _, num_state_dim = self.C.shape
        num_traces, num_steps = self.z.shape

        # init kalman placeholders
        kalman_gain_mats = np.zeros([num_traces, num_steps + 1, num_state_dim, num_state_dim])
        kalman_gain_vecs = np.zeros([num_traces, num_steps + 1, num_state_dim])

        # x_0 ~ N(0, P0) has information matrix P0^{-1}, not P0.
        # Eq. (181) of Fox et al. (arXiv:1003.3829).
        prior_precision = np.linalg.inv(self.latent_prior)
        for d in range(num_traces):
            kalman_gain_mats[d, 0] = prior_precision
            kalman_gain_vecs[d, 0] = 0

        # C^T R^{-1} and C^T R^{-1} C, cached on the model.
        mat_c, mat_inde = self.emission_information()

        # forward iterative
        for d in range(num_traces):
            for t in range(1, num_steps + 1):
                z_t = self.z[d, t - 1]
                dyn_mat = self.dynamics.A[z_t]
                dyn_var_mat = self.dynamics.Sigma[z_t]
                inv_dyn_mat = np.linalg.inv(dyn_mat)
                inv_dyn_var_mat = np.linalg.inv(dyn_var_mat)

                # get first of all m
                m = inv_dyn_mat.transpose() @ kalman_gain_mats[d, t - 1] @ inv_dyn_mat
                j = m @ np.linalg.inv(m + inv_dyn_var_mat)
                complement = np.eye(num_state_dim) - j

                kalman_gain_mats[d, t] = (
                    complement @ m @ complement.transpose() + j @ inv_dyn_var_mat @ j.transpose()
                )
                kalman_gain_vecs[d, t] = (
                    complement @ inv_dyn_mat.transpose() @ kalman_gain_vecs[d, t - 1]
                )

                # appedn extra terms
                kalman_gain_mats[d, t] += mat_inde
                kalman_gain_vecs[d, t] += mat_c @ self.y[d, t - 1]

        return kalman_gain_mats, kalman_gain_vecs

    def slds_backward_kalman_filter_gains(self):
        """Use the numerically stable backward kalman information filter.
        Provided by E. Fox in the report 1003.3829.

        :param conf: The configuration to use.
        :param mode_seq: The truncated mode sequence [DxT] of maximum L used modes.
        :param observations: The observation matrix [DxTxO] where each column is one time step.
        :return: The kalman matrices Lambda [Dx(T+1)xHxH] and theta [Dx(T+1)xH] computed throughout.
        """

        # rename a little bit
        _, num_state_dim = self.C.shape
        num_traces, num_steps = self.z.shape

        # init kalman placeholders
        kalman_gain_mats = np.zeros([num_traces, num_steps + 1, num_state_dim, num_state_dim])
        kalman_gain_vecs = np.zeros([num_traces, num_steps + 1, num_state_dim])

        # init them
        mat_c, mat_inde = self.emission_information()

        # fill all matrices
        for d in range(num_traces):
            kalman_gain_mats[d, num_steps] = mat_inde
            kalman_gain_vecs[d, num_steps] = mat_c @ self.y[d, num_steps - 1]

        # iterate backward
        for d in range(num_traces):
            for t in reversed(range(num_steps)):
                # get mode
                z_nt = self.z[d, t]
                dyn_mat = self.dynamics.A[z_nt]
                dyn_var_mat = self.dynamics.Sigma[z_nt]
                inv_dyn_var_mat = np.linalg.inv(dyn_var_mat)

                # first of all j_hat
                j_hat = kalman_gain_mats[d, t + 1] @ np.linalg.inv(
                    kalman_gain_mats[d, t + 1] + inv_dyn_var_mat
                )
                l_hat = np.eye(num_state_dim) - j_hat

                # calc matrix
                inner_term = (
                    l_hat @ kalman_gain_mats[d, t + 1] @ l_hat.transpose()
                    + j_hat @ inv_dyn_var_mat @ j_hat.transpose()
                )
                kalman_gain_mats[d, t] = dyn_mat.transpose() @ inner_term @ dyn_mat
                kalman_gain_vecs[d, t] = dyn_mat.transpose() @ l_hat @ kalman_gain_vecs[d, t + 1]

                # save
                if t > 0:
                    kalman_gain_mats[d, t] += mat_inde
                    kalman_gain_vecs[d, t] += mat_c @ self.y[d, t - 1]

        return kalman_gain_mats, kalman_gain_vecs
