"""Typed parameter state, the model sweep, and the vectorized updates."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from model.state import DynamicsState, StickyHDPState, load_latent_and_emission_scale
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel
from slds.regression import split_regression_pairs
from slds.slds_ard_prior_sample import _ard_normal_equations
from tests.test_sampler import _model


def test_state_objects_keep_the_named_fields() -> None:
    hdp = StickyHDPState.from_sequence([1.5, 0.25, 3.0, np.array([0.2, 0.8]), np.eye(2)])
    dynamics = DynamicsState.from_sample(
        [np.zeros((2, 2, 2)), np.ones((2, 2, 2)), np.full((2, 2), 4.0)]
    )
    assert hdp.alpha_plus_kappa == 1.5
    assert hdp.rho == 0.25
    assert hdp.pi.shape == (2, 2)
    assert dynamics.alpha is not None
    assert dynamics.alpha.shape == (2, 2)


def test_load_accepts_honest_prior_filenames() -> None:
    # Different latent and observation sizes make the two filename
    # conventions distinguishable. Equal sizes stay on the historical swap.
    model = _model(ld=2, od=1)
    model.hdp = StickyHDPState(1.0, 0.4, 2.0, np.array([0.3, 0.7]), np.full((2, 2), 0.5))
    model.dynamics = DynamicsState(np.eye(2)[None, :, :], np.eye(2)[None, :, :])
    model.R = 0.2 * np.eye(2)
    model.z = np.zeros((1, 2), dtype=np.int64)
    model.x = np.zeros((1, 2, 2))
    model.e = np.zeros((1, 2, 2))
    model.y = np.zeros((1, 2, 2))
    model.w = np.zeros((1, 2, 2))
    with tempfile.TemporaryDirectory() as folder:
        model.store(folder)
        # Historical store writes the emission scale to P0.npy. Swap the
        # contents so the names match the quantities.
        latent = np.load(os.path.join(folder, "R0.npy"))
        emission = np.load(os.path.join(folder, "P0.npy"))
        np.save(os.path.join(folder, "P0.npy"), latent)
        np.save(os.path.join(folder, "R0.npy"), emission)
        loaded = StickyHDPSLDSHMMModel.load(folder)
        assert np.allclose(loaded.latent_prior, model.latent_prior)
        assert np.allclose(loaded.emission_prior[1], model.emission_prior[1])
        direct_latent, direct_emission = load_latent_and_emission_scale(folder, model.ld, model.od)
        assert np.allclose(direct_latent, model.latent_prior)
        assert np.allclose(direct_emission, model.emission_prior[1])


def test_gibbs_step_can_leave_the_emission_fixed() -> None:
    np.random.seed(4)
    model = _model()
    model.init_params()
    model.simulate(2, 6, init=False)
    frozen = np.array(model.R, copy=True)
    model.gibbs_step(sample_r=False)
    assert np.allclose(model.R, frozen)
    assert np.all(np.isfinite(model.x))
    assert np.all(np.isfinite(model.z))


def test_emission_information_is_cached_until_r_changes() -> None:
    model = _model()
    model.R = 0.4 * np.eye(2)
    model.C = np.eye(2)
    first, second = model.emission_information()
    again, _ = model.emission_information()
    assert again is first
    expected = model.C.T @ np.linalg.inv(model.R) @ model.C
    assert np.allclose(second, expected)
    model.R = 0.8 * np.eye(2)
    _, refreshed = model.emission_information()
    assert not np.allclose(refreshed, second)


def test_ard_normal_equations_match_the_column_major_loop() -> None:
    rng = np.random.default_rng(2)
    latent = 3
    count = 5
    parents = rng.normal(size=(latent, count))
    children = rng.normal(size=(latent, count))
    precision = rng.normal(size=(latent, latent))
    precision = precision @ precision.T + np.eye(latent)
    column_precision = np.array([0.5, 1.5, 2.5])
    info, gram = _ard_normal_equations(parents, children, precision, column_precision)

    reference_info = np.zeros(latent**2)
    reference_gram = np.diag(np.repeat(column_precision, latent))
    for index in range(count):
        design = np.tile(np.eye(latent), [1, latent])
        design *= np.expand_dims(np.repeat(parents[:, index], latent), 0)
        reference_info += design.T @ precision @ children[:, index]
        reference_gram += design.T @ precision @ design
    assert np.allclose(info, reference_info)
    assert np.allclose(gram, reference_gram)


def test_shared_splitter_matches_pair_counts() -> None:
    modes = np.array([[0, 1, 1], [1, 0, 0]])
    states = np.arange(12, dtype=float).reshape(2, 3, 2)
    parents, children, counts = split_regression_pairs(modes, states, 2)
    assert list(counts) == [2, 2]
    assert children[1].shape == (2, 2)
    assert np.allclose(parents[1][:, 0], states[0, 0])
    assert np.allclose(children[1][:, 0], states[0, 1])
