# Sticky HDP-SLDS

Gibbs sampling for a sticky hierarchical Dirichlet process switching linear dynamical system, following Fox, Sudderth, Jordan, and Willsky, [Bayesian Nonparametric Inference of Switching Linear Dynamical Systems](https://arxiv.org/abs/1003.3829).

The sampler alternates between:

- block sampling of the latent state with a backward information-form Kalman filter
- block and sequential sampling of the mode sequence
- sticky HDP updates for the transition rows, the global weights, and the concentration parameters
- matrix-normal inverse-Wishart or automatic relevance determination updates for the mode dynamics
- an inverse-Wishart update for the shared emission covariance

Parameters live on two objects. `StickyHDPState` holds `alpha_plus_kappa`, `rho`, `gamma`, `beta`, and `pi`. `DynamicsState` holds the per-mode maps `A`, the process covariances `Sigma`, and the optional ARD precisions `alpha`. One call, `StickyHDPSLDSHMMModel.gibbs_step`, owns the sweep order.

The [install page](install.md) covers `uv`. The [results](results.md) page records the pendulum fit and the 80-step forecast. The [refactor analysis](REFACTOR_ANALYSIS.md) is the design note for the sampler corrections and the remaining numerical-convention work.
