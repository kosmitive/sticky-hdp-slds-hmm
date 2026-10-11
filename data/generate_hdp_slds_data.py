from os.path import join

import numpy as np
from misc.output import print_line
from model.ard_prior import ARDPrior
from model.mniw_prior import MNIWPrior
from model.sticky_hdp_prior import StickyHDPPrior
from model.sticky_hdp_slds_hmm import StickyHDPSLDSHMMModel

from data.generate_hdp_slds_groundtruth import generate_hdp_slds_groundtrouth

# Recorded seed for the checked-in packs. Each pack is drawn from this seed
# on its own, so regenerating one pack does not depend on the other.
DATA_PACK_SEED = 0


def _record_seed(folder):
    with open(join(folder, "SEED"), "w", encoding="utf-8") as handle:
        handle.write(f"{DATA_PACK_SEED}\n")


def generate_mniw_data(root="./data_packs", name="hdp_slds_mniw"):

    print_line()
    print_line("Generate HDP-SLDS-HMM using MNIW prior")
    np.random.seed(DATA_PACK_SEED)

    # some general settings
    num_traces = 50
    num_steps = 100
    ld = 3
    od = 2
    max_z = 5

    # projection
    C = np.hstack((np.eye(od), np.zeros([od, ld - od])))

    # setup hdp prior
    alpha_kappa0 = [15.0, 3.0]
    rho0 = [5.0, 2.0]
    gamma0 = [10.0, 2.0]

    # setup dyn prior
    M = np.zeros([ld] * 2)
    K = 0.5 * np.eye(ld)
    n0 = 10
    S0 = 0.1 * np.eye(ld)

    # remaining priors
    P0 = 1e-6 * np.eye(ld)
    r0 = 3
    R0 = 1e-6 * np.eye(od)

    # create priors
    hdp_prior = StickyHDPPrior(max_z, alpha_kappa0, rho0, gamma0, "uniform")
    mniw_prior = MNIWPrior(M, K, n0, S0)
    latent_prior = P0
    emission_prior = [r0, R0]

    # final model
    folder = join(root, name)
    model = StickyHDPSLDSHMMModel(ld, od, hdp_prior, mniw_prior, latent_prior, emission_prior, C)
    generate_hdp_slds_groundtrouth(folder, num_traces, num_steps, model)
    _record_seed(folder)


def generate_ard_data(root="./data_packs", name="hdp_slds_ard"):

    print_line()
    print_line("Generate HDP-SLDS-HMM using ARD prior")
    np.random.seed(DATA_PACK_SEED)

    # some general settings
    num_traces = 50
    num_steps = 100
    ld = 3
    od = 2
    max_z = 5

    # projection
    C = np.hstack((np.eye(od), np.zeros([od, ld - od])))

    # setup hdp prior
    alpha_kappa0 = [15.0, 3.0]
    rho0 = [5.0, 2.0]
    gamma0 = [10.0, 2.0]

    # setup dyn prior
    alpha0 = [16.0, 8.0]
    n0 = 5
    S0 = 2 * np.eye(ld)

    # remaining priors
    P0 = 1e-6 * np.eye(ld)
    r0 = 3
    R0 = 1e-6 * np.eye(od)

    # create priors
    hdp_prior = StickyHDPPrior(max_z, alpha_kappa0, rho0, gamma0, "uniform")
    ard_prior = ARDPrior(alpha0, n0, S0)
    latent_prior = P0
    emission_prior = [r0, R0]

    # final model
    folder = join(root, name)
    model = StickyHDPSLDSHMMModel(ld, od, hdp_prior, ard_prior, latent_prior, emission_prior, C)
    generate_hdp_slds_groundtrouth(folder, num_traces, num_steps, model)
    _record_seed(folder)


if __name__ == "__main__":
    generate_ard_data()
    generate_mniw_data()
