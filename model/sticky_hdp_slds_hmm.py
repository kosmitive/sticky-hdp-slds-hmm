from os.path import join

import numpy as np
from misc.disc_sample import sample_cat
from misc.hdd_save_load import load_json, save_json
from scipy.stats import invwishart
from slds.densities import gaussian_logpdf
from slds.slds_ard_prior_sample import StickyHDPSLDSHMMModelARDMixin
from slds.slds_block_sample_x import StickyHDPSLDSHMMModelBlockXMixin
from slds.slds_block_sample_z import StickyHDPSLDSHMMModelBlockZMixin
from slds.slds_init import StickyHDPSLDSHMMModelInitMixin
from slds.slds_kalman_filter_y import StickyHDPSLDSHMMModelKalmanYMixin
from slds.slds_mniw_prior_sample import StickyHDPSLDSHMMModelMNIWMixin
from slds.slds_sample_r import StickyHDPSLDSHMMModelEmissionMixin
from slds.slds_sample_sticky_hdp_params import StickyHDPSLDSHMMModelHDPMixin
from slds.slds_seq_sample_z import StickyHDPSLDSHMMModelSeqZMixin

from model.ard_prior import ARDPrior
from model.mniw_prior import MNIWPrior
from model.state import DynamicsState, StickyHDPState, load_latent_and_emission_scale
from model.sticky_hdp_prior import StickyHDPPrior


class StickyHDPSLDSHMMModel(
    StickyHDPSLDSHMMModelBlockXMixin,
    StickyHDPSLDSHMMModelBlockZMixin,
    StickyHDPSLDSHMMModelKalmanYMixin,
    StickyHDPSLDSHMMModelMNIWMixin,
    StickyHDPSLDSHMMModelEmissionMixin,
    StickyHDPSLDSHMMModelHDPMixin,
    StickyHDPSLDSHMMModelSeqZMixin,
    StickyHDPSLDSHMMModelInitMixin,
    StickyHDPSLDSHMMModelARDMixin,
):
    """Sticky HDP switching linear dynamical system.

    The Gibbs conditionals live on the mixins in ``slds``. ``hdp`` is a
    ``StickyHDPState`` and ``dynamics`` is a ``DynamicsState``. ``C`` is the
    fixed emission matrix. ``latent_prior`` is the covariance of the
    auxiliary parent state, and ``emission_prior`` is
    ``[degrees_of_freedom, scale]`` for the shared measurement covariance.
    """

    fn_main = "hdp_slds_hmm_model.json"
    fn_beta = "beta.npy"
    fn_pi = "pi.npy"
    fn_R0 = "P0.npy"
    fn_ard_alpha = "ard_alpha.npy"
    fn_P0 = "R0.npy"
    fn_A = "A.npy"
    fn_Sigma = "Sigma.npy"
    fn_C = "C.npy"
    fn_R = "R.npy"
    fn_z = "z.npy"
    fn_x = "x.npy"
    fn_e = "e.npy"
    fn_y = "y.npy"
    fn_w = "w.npy"

    def __init__(
        self,
        latent_dim,
        observ_dim,
        hdp_prior,
        dyn_prior,
        latent_prior,
        emission_prior,
        emission_proj,
    ):

        # sample some initial parameters
        self.dyn_prior = dyn_prior
        self.hdp_prior = hdp_prior
        self.latent_prior = latent_prior
        self.emission_prior = emission_prior
        self.ld = latent_dim
        self.od = observ_dim

        # sample
        self.dynamics = None
        self.hdp = None
        self._emission_info = None
        self._emission_info_token = None
        self.C = emission_proj
        self.R = None
        self.z = None
        self.x = None
        self.e = None
        self.y = None
        self.w = None

        # reserve space for
        self.hist_alpha_kappa = list()
        self.hist_rho = list()
        self.hist_gamma = list()

        # dispatch
        if isinstance(dyn_prior, ARDPrior):
            self.dyn_prior_type = "ard"
        elif isinstance(dyn_prior, MNIWPrior):
            self.dyn_prior_type = "mniw"
        else:
            self.dyn_prior_type = None

    def calc_log_likelihood(self) -> float:
        """Joint log density of the observations, latents, and mode sequence.

        The initial mode is scored under ``beta``. Each later mode is scored
        under the corresponding row of ``pi``. Latent transitions start at the
        second time index, because the first stored state has no parent.

        Returns:
            The log joint, up to terms that do not depend on the sampled variables.
        """

        state_transition_ll = self.__calc_mode_transition_ll()
        latent_transition_ll = self.__calc_latent_transition_ll()
        emission_ll = self.__calc_emission_ll()
        return emission_ll + state_transition_ll + latent_transition_ll

    def __calc_latent_transition_ll(self) -> float:
        """Log density of ``x_t | x_{t-1}, z_t`` for every trace and ``t >= 1``."""

        if self.x.shape[1] < 2:
            return 0.0
        parents = self.x[:, :-1]
        children = self.x[:, 1:]
        modes = self.z[:, 1:]
        latent_transition_ll = 0.0
        for mode in range(self.dynamics.A.shape[0]):
            selected = modes == mode
            if not np.any(selected):
                continue
            mean = parents[selected] @ self.dynamics.A[mode].T
            latent_transition_ll += float(
                np.sum(gaussian_logpdf(children[selected], mean, self.dynamics.Sigma[mode]))
            )
        return latent_transition_ll

    def __calc_mode_transition_ll(self) -> float:
        """Log density of the mode chain, including the ``beta`` draw of ``z_0``."""

        pi = np.asarray(self.hdp.pi, dtype=np.float64)
        beta = np.asarray(self.hdp.beta, dtype=np.float64)
        num_modes = pi.shape[0]
        safe_pi = np.clip(pi, 1e-12, None)
        safe_pi = safe_pi / np.sum(safe_pi, axis=1, keepdims=True)

        transitions = (num_modes * self.z[:, :-1] + self.z[:, 1:]).reshape(-1)
        transition_factors = np.bincount(transitions, minlength=num_modes**2)
        transition_count_matrix = transition_factors.reshape([num_modes] * 2)
        state_transition_ll = np.sum(transition_count_matrix * np.log(safe_pi))
        state_transition_ll += np.sum(np.log(np.clip(beta[self.z[:, 0]], 1e-12, None)))
        return float(state_transition_ll)

    def __calc_emission_ll(self):
        """Calculate log likelihood of emissions."""

        mean_y = self.x.reshape(-1, self.ld) @ self.C.T
        observations = self.y.reshape(-1, self.od)
        return float(np.sum(gaussian_logpdf(observations, mean_y, self.R)))

    def emission_information(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``C^T R^{-1}`` and ``C^T R^{-1} C``, cached until either changes.

        Returns:
            The cross information and the state information from one emission.
        """

        token = (id(self.C), id(self.R))
        if token != self._emission_info_token:
            precision = np.linalg.inv(self.R)
            cross = self.C.T @ precision
            self._emission_info = (cross, cross @ self.C)
            self._emission_info_token = token
        return self._emission_info

    def simulate(self, num_traces: int, num_steps: int, init: bool = True) -> None:
        """Simulate traces and store every series except the auxiliary parent state.

        Args:
            num_traces: Number of independent sequences.
            num_steps: Number of stored time steps per sequence.
            init: When true, redraw HDP, dynamics, and emission parameters
                from their priors before simulating. Set this false to keep
                parameters that have already been assigned.
        """

        # reserve space for the simulation
        z = np.empty([num_traces, num_steps + 1], dtype=np.int64)
        x = np.empty([num_traces, num_steps + 1, self.ld])

        if init or self.dynamics is None or self.hdp is None or self.hdp.pi is None:
            self.init_params()
        beta = self.hdp.beta
        pi = self.hdp.pi

        # generate the emission noise before hand
        e = np.empty([num_traces, num_steps, self.ld])

        # repeat till all traces were generated
        for t in range(num_traces):
            # Simply choose one z from beta and draw a random multivariate x
            z[t, 0] = sample_cat(beta)
            x[t, 0] = np.random.multivariate_normal(np.zeros(self.ld), self.latent_prior)

            # execute all steps
            for s in range(1, num_steps + 1):
                # sample new mode
                pre_z = z[t, s - 1]
                cur_z = sample_cat(pi[pre_z])

                # get the current mode as an integer
                e[t, s - 1] = np.random.multivariate_normal(
                    np.zeros(self.ld), self.dynamics.Sigma[cur_z]
                )
                x[t, s] = self.dynamics.A[cur_z] @ x[t, s - 1] + e[t, s - 1]
                z[t, s] = cur_z

        # emissions. Drop the same initial time index that is dropped from y.
        w = np.random.multivariate_normal(np.zeros(self.od), self.R, [num_traces, num_steps + 1])
        y = x.reshape([-1, self.ld]) @ self.C.transpose() + w.reshape([-1, self.od])
        y = y.reshape([num_traces, num_steps + 1, self.od])

        self.z = z[:, 1:]
        self.x = x[:, 1:]
        self.e = e
        self.y = y[:, 1:]
        self.w = w[:, 1:]

    def init_params(self):
        """Draw HDP, dynamics, and emission parameters from their priors."""

        self.dynamics = DynamicsState.from_sample(self.dyn_prior.prior_sample(self.hdp_prior.max_z))
        self.hdp = StickyHDPState.from_sequence(self.hdp_prior.max_likely_sample())
        self.R = invwishart.rvs(self.emission_prior[0], self.emission_prior[1])
        self._emission_info_token = None

    def gibbs_step(
        self,
        sample_block_z: bool = True,
        sample_seq_z: bool = True,
        sample_x: bool = True,
        sample_hdp_alpha_kappa: bool = True,
        sample_hdp_rho: bool = True,
        sample_hdp_gamma: bool = True,
        sample_hdp_beta: bool = True,
        sample_hdp_pi: bool = True,
        sample_dynamics: bool = True,
        sample_r: bool = True,
        rounds_hmm: int = 1,
        rounds_hdp: int = 1,
        rounds_ard: int = 1,
    ) -> None:
        """Run one Gibbs sweep in a fixed order.

        Modes are updated first, then the information filters, then the latent
        states, then the sticky HDP, the dynamics, and the emission covariance.
        Flags omit a block. ``rounds_hmm``, ``rounds_hdp``, and ``rounds_ard``
        repeat the mode-and-state block, the sticky HDP block, and the ARD
        dynamics block.

        Args:
            sample_block_z: Draw the whole mode path given the latents.
            sample_seq_z: Draw each mode given the others.
            sample_x: Draw the latent path given the modes.
            sample_hdp_alpha_kappa: Draw ``alpha + kappa``.
            sample_hdp_rho: Draw the sticky weight.
            sample_hdp_gamma: Draw the top-level concentration.
            sample_hdp_beta: Draw the global weights.
            sample_hdp_pi: Draw the transition rows.
            sample_dynamics: Draw the linear maps and process covariances.
            sample_r: Draw the shared emission covariance.
            rounds_hmm: Repeats of the mode, filter, and state block.
            rounds_hdp: Repeats of the sticky HDP block.
            rounds_ard: Repeats of the ARD dynamics update.
        """

        for _ in range(rounds_hmm):
            if sample_block_z:
                self.slds_post_block_sample_z()
            if sample_seq_z or sample_x:
                backward = self.slds_backward_kalman_filter_gains()
                if sample_seq_z:
                    forward = self.slds_forward_kalman_filter_gains()
                    self.slds_post_seq_sample_z(forward[0], forward[1], backward[0], backward[1])
                    backward = self.slds_backward_kalman_filter_gains()
                if sample_x:
                    self.slds_post_block_sample_x(backward[0], backward[1])

        sticky = (
            sample_hdp_alpha_kappa
            or sample_hdp_rho
            or sample_hdp_gamma
            or sample_hdp_beta
            or sample_hdp_pi
        )
        for _ in range(rounds_hdp):
            if sticky:
                self.calc_transitions()
            if sample_hdp_beta or sample_hdp_alpha_kappa or sample_hdp_gamma or sample_hdp_rho:
                self.sample_post_auxiliary_vars()
            if sample_hdp_alpha_kappa:
                self.sample_post_hdp_alpha_kappa()
            if sample_hdp_rho:
                self.sample_post_hdp_rho()
            if sample_hdp_gamma:
                self.sample_post_hdp_gamma()
            if sample_hdp_beta:
                self.sample_post_hdp_beta()
            if sample_hdp_pi:
                self.sample_post_hdp_pi()

        if sample_dynamics:
            if self.dyn_prior_type == "ard":
                for _ in range(rounds_ard):
                    self.slds_post_ard_dynamics_param_sampling()
            else:
                self.slds_post_mniw_dynamics_param_sampling()
        if sample_r:
            self.slds_post_sample_r()

    @staticmethod
    def load(folder):
        """Load the prior from the specified folder."""

        # reload data
        data = load_json(folder, StickyHDPSLDSHMMModel.fn_main)
        hdp_prior = StickyHDPPrior.load(folder)

        # depending on the type of prior
        dyn_prior = ARDPrior.load(folder) if data["prior"] == "ard" else MNIWPrior.load(folder)

        # save prior matrices
        C = np.load(join(folder, StickyHDPSLDSHMMModel.fn_C))
        ld = data["ld"]
        od = data["od"]
        latent_prior, emission_scale = load_latent_and_emission_scale(folder, ld, od)
        emission_prior = [data["r0"], emission_scale]

        # create object
        m = StickyHDPSLDSHMMModel(ld, od, hdp_prior, dyn_prior, latent_prior, emission_prior, C)

        # load hdp
        beta = np.load(join(folder, StickyHDPSLDSHMMModel.fn_beta))
        pi = np.load(join(folder, StickyHDPSLDSHMMModel.fn_pi))
        m.hdp = StickyHDPState(data["alpha_kappa"], data["rho"], data["gamma"], beta, pi)

        # load dynamics
        dynamics_alpha = None
        if data["prior"] == "ard":
            dynamics_alpha = np.load(join(folder, StickyHDPSLDSHMMModel.fn_ard_alpha))
        m.dynamics = DynamicsState(
            np.load(join(folder, StickyHDPSLDSHMMModel.fn_A)),
            np.load(join(folder, StickyHDPSLDSHMMModel.fn_Sigma)),
            dynamics_alpha,
        )

        # save rest
        m.R = np.load(join(folder, StickyHDPSLDSHMMModel.fn_R))
        m.z = np.load(join(folder, StickyHDPSLDSHMMModel.fn_z))
        m.x = np.load(join(folder, StickyHDPSLDSHMMModel.fn_x))
        m.e = np.load(join(folder, StickyHDPSLDSHMMModel.fn_e))
        m.y = np.load(join(folder, StickyHDPSLDSHMMModel.fn_y))
        m.w = np.load(join(folder, StickyHDPSLDSHMMModel.fn_w))

        return m

    def store(self, folder):
        """Store the prior in the specified folder."""

        # store the dict
        prior_type = "ard" if isinstance(self.dyn_prior, ARDPrior) else "mniw"
        data = {
            "ld": self.ld,
            "od": self.od,
            "prior": prior_type,
            "r0": self.emission_prior[0],
            "alpha_kappa": self.hdp.alpha_plus_kappa,
            "rho": self.hdp.rho,
            "gamma": self.hdp.gamma,
        }
        save_json(folder, StickyHDPSLDSHMMModel.fn_main, data)

        # sample some initial parameters
        self.dyn_prior.store(folder)
        self.hdp_prior.store(folder)

        # save prior matrices
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_R0), self.emission_prior[1])
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_P0), self.latent_prior)

        # main parameters
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_A), self.dynamics.A)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_Sigma), self.dynamics.Sigma)

        if prior_type == "ard":
            np.save(join(folder, StickyHDPSLDSHMMModel.fn_ard_alpha), self.dynamics.alpha)

        np.save(join(folder, StickyHDPSLDSHMMModel.fn_C), self.C)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_R), self.R)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_z), self.z)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_x), self.x)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_e), self.e)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_y), self.y)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_w), self.w)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_beta), self.hdp.beta)
        np.save(join(folder, StickyHDPSLDSHMMModel.fn_pi), self.hdp.pi)
