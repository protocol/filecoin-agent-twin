#!/usr/bin/env python3

"""
We have two experiments here
Baseline - FIL+ and CC NPV agents, with a proportional power distribution.
Experiment - FIL+, CC, and Risk Averse CC NPV agents, with a proportional power distribution.
  As we sweep the percentage of risk-averse NPV agents, what happens to the network KPI?
"""
import argparse
from typing import Dict, Tuple, List
from datetime import date

import numpy as np

def generate_experiments(output_fp):
    print('Writing to {}'.format(output_fp))

    experiment_names = []

    max_daily_rb_onboard_pib_vec = [4, 6, 8]
    renewal_rate_vec = [0.6, 0.7, 0.8]
    fil_plus_rate_vec = [.4, .6, .8]
    sector_duration = 360
    for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for fil_plus_rate in fil_plus_rate_vec:
                name = 'BaselineDCA_RBP_%0.02f-RR_%0.02f-FPR_%0.02f-Dur_%0.02f' % \
                    (max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)


if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='baseline.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_experiments(output_fp)
