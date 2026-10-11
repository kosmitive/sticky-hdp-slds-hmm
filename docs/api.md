# API

The public container is `StickyHDPSLDSHMMModel`. Gibbs conditionals stay on the mixins in `slds`. Call `gibbs_step` rather than ordering those mixins at the call site.

## Parameter state

::: model.state.StickyHDPState

::: model.state.DynamicsState

## Model

::: model.sticky_hdp_slds_hmm.StickyHDPSLDSHMMModel
    options:
      members:
        - gibbs_step
        - calc_log_likelihood
        - simulate
        - init_params
        - emission_information
        - load
        - store
