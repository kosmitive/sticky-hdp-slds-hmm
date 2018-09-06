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


import os
from os.path import join

from debug.log import create_logtoken
# sampling imports
from misc.history import History
from misc.output import print_line, gen_plot, print_dbl_line
from model.gt_sticky_hdp_slds_hmm import StickyHDPSLDSHMMConfiguration


def exec_test(l_token, test_conf):

    # print headers
    print_line()
    print("Test '{}' started".format(test_conf['test_name']))

    # get the root folder
    gibbs_steps = test_conf['gibbs_steps']

    # init for test
    m = StickyHDPSLDSHMMConfiguration.init_for_test(test_conf)
    max_z = m.gt.hdp_prior.max_z

    # define the history fields
    print_out_steps = test_conf['print_out_steps']
    res_path = l_token.init_test(test_conf['test_name'])

    for cs in range(gibbs_steps):

        m.gibbs_step()
        m.log_errors()

        if cs % print_out_steps == 0:

            print_line()
            print_line("Iteration {}".format(cs))
            print_line()

            m.print_model(print_out_prior=test_conf['print_out_prior'],
                          num_traces=test_conf['print_traces'])

    m.store_run(res_path)


def execute_testcase_defs(testrun_conf, testcase_defs):

    # print headers
    print_dbl_line()
    print("Tests started...")

    # output root folder
    root_folder = join(testrun_conf['root'])
    l_token = create_logtoken(root_folder)

    # iterate over all testcase definitions.
    for test_conf in testcase_defs:
        exec_test(l_token, {**test_conf, **testrun_conf})
