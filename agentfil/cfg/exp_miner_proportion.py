#!/usr/bin/env python3

"""
# TODO: describe the experiments
"""
import argparse
from typing import Dict, Tuple, List
from datetime import date

import numpy as np


def generate_miner_proportion_network_sensitivity_experiments(output_fp):
    print('Writing to {}'.format(output_fp))

    experiment_names = []
    total_daily_rb_onboard_pib_vec = [4, 6, 8]
    renewal_rate_vec = [0.5, 0.6, 0.7]
    filp_agent_power_distribution_vec = np.arange(0.3, 0.7+0.1, 0.1)
    
    fil_supply_discount_rate_vec = [10, 20, 30]
    filplus_agent_optimism_vec = [4]
    normal_cc_agent_optimism_vec = [4]
    
    filplus_agent_discount_rate_yr_pct_vec = [25, 50]
    normal_cc_agent_discount_rate_multiplier_vec = [1, 2]
    
    for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for filp_agent_power in filp_agent_power_distribution_vec:
                for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                    for filplus_agent_optimism in filplus_agent_optimism_vec:
                        for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                            for filplus_agent_discount_rate in filplus_agent_discount_rate_yr_pct_vec:
                                for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                    normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * filplus_agent_discount_rate
                                    
                                    name = 'MinerProportionSensitivity,FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                        (
                                            filplus_agent_optimism, filplus_agent_discount_rate, filp_agent_power,
                                            normal_cc_agent_optimism, normal_cc_agent_discount_rate, 
                                            total_daily_rb_onboard_pib, renewal_rate, fil_supply_discount_rate,
                                        )
                                    experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)


if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='sdm_experiments.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_miner_proportion_network_sensitivity_experiments(output_fp)