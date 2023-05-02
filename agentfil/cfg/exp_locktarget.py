#!/usr/bin/env python3

"""
TODO: explain
"""
import argparse
from typing import Dict, Tuple, List
from datetime import date

import numpy as np

from agentfil.filecoin_model import FilecoinModel

def lock_target_linear_ramp(filecoin_model: FilecoinModel, 
                     new_lock_target_value = 0.3, 
                     new_lock_target_start_date = date(2023, 10, 15),
                     new_lock_target_end_date = date(2023, 12, 15)):  # ramp over two months, by default
    # map lock-target values in the date-ranges
    num_days_ramp = (new_lock_target_end_date - new_lock_target_start_date).days
    lock_target_delta_per_day = (new_lock_target_value - filecoin_model.lock_target) / num_days_ramp
    if filecoin_model.current_date >= new_lock_target_start_date and filecoin_model.current_date <= new_lock_target_end_date:
        filecoin_model.lock_target += lock_target_delta_per_day

def lock_target_jump(filecoin_model: FilecoinModel, 
                     new_lock_target_value = 0.3, 
                     new_lock_target_start_date = date(2023, 10, 15),
                     new_lock_target_end_date = None):
    # new_lock_target_end_date --> ignored in the jump variant
    if filecoin_model.current_date >= new_lock_target_start_date:
        filecoin_model.lock_target = new_lock_target_value

def get_lock_target_post_step_callable(lock_target_increase_dynamics: str, 
                                       lock_target_increase_value: float,
                                       lock_target_increase_date_start: date,
                                       lock_target_increase_date_end: date):
    kwargs = {
        'new_lock_target_value': lock_target_increase_value,
        'new_lock_target_start_date': lock_target_increase_date_start,
        'new_lock_target_end_date': lock_target_increase_date_end,
    }
    if lock_target_increase_dynamics == 'linear_ramp':
        return lock_target_linear_ramp, kwargs
    elif lock_target_increase_dynamics == 'jump':
        return lock_target_jump, kwargs
    else:
        raise ValueError('Unknown lock target increase dynamics: {}'.format(lock_target_increase_dynamics))


def generate_experiments(output_fp):
    print('Writing to {}'.format(output_fp))

    experiment_names = []
    total_daily_rb_onboard_pib_vec = [6]
    renewal_rate_vec = [0.6]
    filp_agent_power_distribution_vec = np.arange(0.3, 0.7+0.1, 0.1)
    
    fil_supply_discount_rate_vec = [20, 30]
    filplus_agent_optimism_vec = [4]
    normal_cc_agent_optimism_vec = [4]
    
    filplus_agent_discount_rate_yr_pct_vec = [25, 50]
    normal_cc_agent_discount_rate_multiplier_vec = [1, 2]

    lock_target_increase_value_vec = [0.4, 0.5, 0.6]
    lock_target_increase_dynamics_vec = ['linear_ramp', 'jump']
    lock_target_ramp_speed_months_vec = [3, 6, 12]
    
    for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for filp_agent_power in filp_agent_power_distribution_vec:
                for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                    for filplus_agent_optimism in filplus_agent_optimism_vec:
                        for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                            for filplus_agent_discount_rate in filplus_agent_discount_rate_yr_pct_vec:
                                for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                    normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * filplus_agent_discount_rate

                                    for lock_target_increase_value in lock_target_increase_value_vec:
                                        for lock_target_increase_dynamics in lock_target_increase_dynamics_vec:
                                            if lock_target_increase_dynamics == 'jump':
                                                name = 'LockTarget[%0.1f,%s],FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                                    (
                                                        lock_target_increase_value, lock_target_increase_dynamics,
                                                        filplus_agent_optimism, filplus_agent_discount_rate, filp_agent_power,
                                                        normal_cc_agent_optimism, normal_cc_agent_discount_rate, 
                                                        total_daily_rb_onboard_pib, renewal_rate, fil_supply_discount_rate,
                                                    )
                                                experiment_names.append(name)

                                            elif lock_target_increase_dynamics == 'linear_ramp':
                                                for lock_target_ramp_speed_months in lock_target_ramp_speed_months_vec:
                                                    name = 'LockTarget[%0.1f,%s,%d],FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                                        (
                                                            lock_target_increase_value, lock_target_increase_dynamics, lock_target_ramp_speed_months,
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
    parser.add_argument('--output_fp', type=str, default='lock_target.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_experiments(output_fp)