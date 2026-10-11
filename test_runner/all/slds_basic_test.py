from test_runner.testbench_slds import execute_testcase_defs

# Every experiment starts with all sampling flags off. A case lists the
# blocks it turns on.
_SAMPLE_FLAGS = (
    "sample_x",
    "sample_block_z",
    "sample_seq_z",
    "sample_hdp_alpha_kappa",
    "sample_hdp_rho",
    "sample_hdp_beta",
    "sample_hdp_gamma",
    "sample_hdp_pi",
    "sample_dynamics",
    "sample_r",
)


def _case(name, **enabled):
    """One experiment: ``name`` plus the sampling flags that are on."""

    unknown = set(enabled) - set(_SAMPLE_FLAGS)
    if unknown:
        raise KeyError(f"Unknown sampling flags: {sorted(unknown)}")
    case = {"test_name": name}
    case.update({flag: False for flag in _SAMPLE_FLAGS})
    case.update(enabled)
    return case


def _conf(model_path, gibbs_steps, **extra):
    """Shared runner settings for a data pack and a sweep count."""

    conf = {
        "print_out_prior": True,
        "print_out_nll": True,
        "print_traces": 1,
        "print_out_steps": 1,
        "print_out_statistics": True,
        "gibbs_steps": gibbs_steps,
        "root": "./test_runs/",
        "model_path": model_path,
    }
    conf.update(extra)
    return conf


def slds_ard_all_test():
    cases = [
        _case(
            "test_ard_all",
            sample_x=True,
            sample_block_z=True,
            sample_seq_z=True,
            sample_dynamics=True,
        ),
        _case("test_block_x", sample_x=True),
        _case("test_block_z", sample_block_z=True),
        _case("test_seq_z", sample_block_z=True, sample_seq_z=True),
    ]
    execute_testcase_defs(_conf("./data_packs/hdp_slds_ard", 100), cases)


def slds_x_z_test():
    cases = [
        _case("test_seq_z", sample_block_z=True, sample_seq_z=True),
        _case("test_block_z", sample_block_z=True),
        _case("test_block_x", sample_x=True),
    ]
    execute_testcase_defs(_conf("./data_packs/hdp_slds_ard", 1000), cases)


def slds_hdp_test():
    cases = [
        _case(
            "test_hdp_all", sample_hdp_alpha_kappa=True, sample_hdp_rho=True, sample_hdp_gamma=True
        ),
        _case("test_hdp_alpha_kappa", sample_hdp_alpha_kappa=True),
        _case("test_hdp_rho", sample_hdp_rho=True),
        _case("test_hdp_beta", sample_hdp_beta=True),
        _case("test_hdp_gamma", sample_hdp_gamma=True),
        _case("test_hdp_pi", sample_hdp_pi=True),
    ]
    execute_testcase_defs(_conf("./data_packs/hdp_slds_ard", 100, rounds_hdp=1), cases)


def slds_dyn_test():
    dynamics = [_case("test_dynamics_ard", sample_dynamics=True)]
    execute_testcase_defs(_conf("./data_packs/hdp_slds_ard", 1000), dynamics)
    # The matrix-normal case is launched against the same pack as the ARD case.
    dynamics = [_case("test_dynamics_mniw", sample_dynamics=True)]
    execute_testcase_defs(_conf("./data_packs/hdp_slds_ard", 1000), dynamics)
