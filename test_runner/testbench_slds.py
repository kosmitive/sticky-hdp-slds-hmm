from os.path import join

from debug.log import create_logtoken

# sampling imports
from misc.output import print_dbl_line, print_line
from model.gt_sticky_hdp_slds_hmm import StickyHDPSLDSHMMConfiguration


def exec_test(l_token, test_conf):

    # print headers
    print_line()
    print("Test '{}' started".format(test_conf["test_name"]))

    # get the root folder
    gibbs_steps = test_conf["gibbs_steps"]

    # init for test
    m = StickyHDPSLDSHMMConfiguration.init_for_test(test_conf)

    # define the history fields
    print_out_steps = test_conf["print_out_steps"]
    res_path = l_token.init_test(test_conf["test_name"])

    for cs in range(gibbs_steps):
        m.gibbs_step()
        m.log_errors()

        if cs % print_out_steps == 0:
            print_line()
            print_line(f"Iteration {cs}")
            print_line()

            m.print_model(
                print_out_prior=test_conf["print_out_prior"], num_traces=test_conf["print_traces"]
            )

    m.store_run(res_path)


def execute_testcase_defs(testrun_conf, testcase_defs):

    # print headers
    print_dbl_line()
    print("Tests started...")

    # output root folder
    root_folder = join(testrun_conf["root"])
    l_token = create_logtoken(root_folder)

    # iterate over all testcase definitions.
    for test_conf in testcase_defs:
        exec_test(l_token, {**test_conf, **testrun_conf})
