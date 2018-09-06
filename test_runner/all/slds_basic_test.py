# MIT License
# 
# Copyright (c) 2018 Markus Semmler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from test_runner.testbench_slds import execute_testcase_defs


def slds_ard_all_test():

    # This represents the test_runner run configuration
    testrun_conf = {

        # print out the only some information
        'print_out_prior': True,
        'print_out_nll': True,
        'print_traces': 1,
        'print_out_steps': 1,
        'print_out_statistics': True,

        'gibbs_steps': 100,
        'root': './test_runs/',
        'model_path': './data_packs/hdp_slds_ard'
    }

    # create a list and fill it appropriately
    testcase_defs = list()
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_ard_all',

        # define which parameters should be estimated
        # and which not.
        'sample_x': True,
        'sample_block_z': True,
        'sample_seq_z': True,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': True,
        'sample_r': False,
    })

    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_block_x',

        # define which parameters should be estimated
        # and which not.
        'sample_x': True,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_block_z',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': True,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_seq_z',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': True,
        'sample_seq_z': True,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })

    # execute
    execute_testcase_defs(testrun_conf, testcase_defs)


def slds_x_z_test():

    # This represents the test_runner run configuration
    testrun_conf = {

        # print out the only some information
        'print_out_prior': True,
        'print_out_nll': True,
        'print_traces': 1,
        'print_out_steps': 1,
        'print_out_statistics': True,

        'gibbs_steps': 1000,
        'root': './test_runs/',
        'model_path': './data_packs/hdp_slds_ard'
    }

    # create a list and fill it appropriately
    testcase_defs = list()
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_seq_z',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': True,
        'sample_seq_z': True,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_block_z',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': True,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_block_x',

        # define which parameters should be estimated
        # and which not.
        'sample_x': True,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })


    # execute
    execute_testcase_defs(testrun_conf, testcase_defs)


def slds_hdp_test():

    # This represents the test_runner run configuration
    testrun_conf = {

        # print out the only some information
        'print_out_prior': True,
        'print_out_nll': True,
        'print_traces': 1,
        'print_out_steps': 1,
        'print_out_statistics': True,
        'rounds_hdp': 1,

        'gibbs_steps': 100,
        'root': './test_runs/',
        'model_path': './data_packs/hdp_slds_ard'
    }

    # create a list and fill it appropriately
    testcase_defs = list()
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_hdp_all',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': True,
        'sample_hdp_rho': True,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': True,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
        'store_hdp_hist': True
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_hdp_alpha_kappa',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': True,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_hdp_rho',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': True,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_hdp_beta',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': True,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_hdp_gamma',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': True,
        'sample_hdp_pi': False,
        'sample_dynamics': False,
        'sample_r': False,
    })
    testcase_defs.append({

        # test_runner settings
        'test_name': 'test_hdp_pi',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': True,
        'sample_dynamics': False,
        'sample_r': False,
    })

    # execute
    execute_testcase_defs(testrun_conf, testcase_defs)


def slds_dyn_test():

    # This represents the test_runner run configuration
    ard_testrun_conf = {

        # print out the only some information
        'print_out_prior': True,
        'print_out_nll': True,
        'print_traces': 1,
        'print_out_steps': 1,
        'print_out_statistics': True,

        'gibbs_steps': 1000,
        'root': './test_runs/',
        'model_path': './data_packs/hdp_slds_ard'
    }

    # create a list and fill it appropriately
    ard_testcase_defs = list()
    ard_testcase_defs.append({

        # test_runner settings
        'test_name': 'test_dynamics_ard',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': True,
        'sample_r': False,
    })

    # execute
    execute_testcase_defs(ard_testrun_conf, ard_testcase_defs)

    # This represents the test_runner run configuration
    mniw_testrun_conf = {

        # print out the only some information
        'print_out_prior': True,
        'print_out_nll': True,
        'print_traces': 1,
        'print_out_steps': 1,
        'print_out_statistics': True,

        'gibbs_steps': 1000,
        'root': './test_runs/',
        'model_path': './data_packs/hdp_slds_ard'
    }

    # create a list and fill it appropriately
    mniw_testcase_defs = list()
    mniw_testcase_defs.append({

        # test_runner settings
        'test_name': 'test_dynamics_mniw',

        # define which parameters should be estimated
        # and which not.
        'sample_x': False,
        'sample_block_z': False,
        'sample_seq_z': False,
        'sample_hdp_alpha_kappa': False,
        'sample_hdp_rho': False,
        'sample_hdp_beta': False,
        'sample_hdp_gamma': False,
        'sample_hdp_pi': False,
        'sample_dynamics': True,
        'sample_r': False,
    })

    # execute
    execute_testcase_defs(mniw_testrun_conf, mniw_testcase_defs)