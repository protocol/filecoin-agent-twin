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

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.npv_agent import NPVAgent

class SDMBaselineExperiment(ExperimentCfg):
    """
    Notes for the experiment:
     1 - We have two agents, FIL+ and CC agents.
     2 - When we bootstrap the agents, they have the provided proportion of power split amongst the
         historical FIL+ and CC sectors.
     3 - Due to this historical seeding, and the 'optimistic' renewals setting which dictates that renewals
         are allowed for FIL+ sectors to simulate the effect of FIL+ sectors getting onboarded at the same
         general rate.
     4 - This is not in the "true" spirit of a FIL+ agent, but a starting approximation.
     5 - Once the simulation starts, the agents will be only FIL+ or CC agents. However, as noted above, renewals
         are allowed for both agents.
    """
    def __init__(self, 
                 max_sealing_throughput,
                 total_daily_onboard_rb_pib, renewal_rate,
                 agent_power_distribution,
                 fil_supply_discount_rate,
                 filplus_agent_optimism, filplus_agent_discount_rate_yr_pct,
                 cc_agent_optimism, cc_agent_discount_rate_yr_pct):
        agent_power_distribution = np.asarray(agent_power_distribution)
        agent_power_distribution = agent_power_distribution / np.sum(agent_power_distribution)  # enforce it to sum to 1

        self.agent_types = [NPVAgent, NPVAgent]
        self.num_agents = len(self.agent_types)
        self.agent_kwargs = [
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': total_daily_onboard_rb_pib * agent_power_distribution[0],
                'renewal_rate': renewal_rate,   
                'fil_plus_rate': 1,  # 100% FIL+ agent
                'agent_optimism': filplus_agent_optimism,
                'agent_discount_rate_yr_pct': filplus_agent_discount_rate_yr_pct,
            },
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': total_daily_onboard_rb_pib * agent_power_distribution[1],
                'renewal_rate': renewal_rate,
                'fil_plus_rate': 0,  # a CC agent
                'agent_optimism': cc_agent_optimism,
                'agent_discount_rate_yr_pct': cc_agent_discount_rate_yr_pct,
            },
        ]
        self.agent_power_distribution = agent_power_distribution

        # external environment configuration
        self.fil_supply_discount_rate = fil_supply_discount_rate

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        return self.agent_types, self.agent_kwargs, self.agent_power_distribution
    
    def get_fil_supply_discount_rate_process_cfg(self):
        fil_supply_discount_rate_process_kwargs = {
            'min_discount_rate_pct':self.fil_supply_discount_rate-1, 
            'max_discount_rate_pct':self.fil_supply_discount_rate+1,
            'start_discount_rate_pct':self.fil_supply_discount_rate,
            'behavior':'constant',
            'seed':1234
        }
        return fil_supply_discount_rate_process_kwargs
    
class SDMExperiment(ExperimentCfg):
    """
    Notes for the experiment:
     1 - We have three agents, FIL+, NormalCC and RiskAverseCC agents.
     2 - When we bootstrap the agents, they have the provided proportion of power split amongst the
         historical FIL+ and CC sectors.
     3 - Due to this historical seeding, and the 'optimistic' renewals setting which dictates that renewals
         are allowed for FIL+ sectors to simulate the effect of FIL+ sectors getting onboarded at the same
         general rate.
     4 - This is not in the "true" spirit of a FIL+ agent, but a starting approximation.
     5 - Once the simulation starts, the agents will be only FIL+ or CC agents. However, as noted above, renewals
         are allowed for both agents.
     6 - Normal and Risk Aversion is controlled by the agent's internal discount rate
    """
    def __init__(self, 
                 max_sealing_throughput,
                 total_daily_onboard_rb_pib, renewal_rate,
                 agent_power_distribution,
                 fil_supply_discount_rate,
                 filplus_agent_optimism, filplus_agent_discount_rate_yr_pct,
                 normal_cc_agent_optimism, normal_cc_agent_discount_rate_yr_pct,
                 riskaverse_cc_agent_optimism, riskaverse_cc_agent_discount_rate_yr_pct):
        agent_power_distribution = np.asarray(agent_power_distribution)
        agent_power_distribution = agent_power_distribution / np.sum(agent_power_distribution)  # enforce it to sum to 1
        
        self.agent_types = [NPVAgent, NPVAgent, NPVAgent]
        self.num_agents = len(self.agent_types)
        self.agent_kwargs = [
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': total_daily_onboard_rb_pib * agent_power_distribution[0],
                'renewal_rate': renewal_rate,   
                'fil_plus_rate': 1,  # 100% FIL+ agent
                'agent_optimism': filplus_agent_optimism,
                'agent_discount_rate_yr_pct': filplus_agent_discount_rate_yr_pct,
            },
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': total_daily_onboard_rb_pib * agent_power_distribution[1],
                'renewal_rate': renewal_rate,
                'fil_plus_rate': 0,  # a CC agent
                'agent_optimism': normal_cc_agent_optimism,
                'agent_discount_rate_yr_pct': normal_cc_agent_discount_rate_yr_pct,
            },
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': total_daily_onboard_rb_pib * agent_power_distribution[2],
                'renewal_rate': renewal_rate,
                'fil_plus_rate': 0,  # a CC agent
                'agent_optimism': riskaverse_cc_agent_optimism,
                'agent_discount_rate_yr_pct': riskaverse_cc_agent_discount_rate_yr_pct,
            },
        ]
        self.agent_power_distribution = agent_power_distribution

        # external environment configuration
        self.fil_supply_discount_rate = fil_supply_discount_rate

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        return self.agent_types, self.agent_kwargs, self.agent_power_distribution
    
    def get_fil_supply_discount_rate_process_cfg(self):
        fil_supply_discount_rate_process_kwargs = {
            'min_discount_rate_pct':self.fil_supply_discount_rate-1, 
            'max_discount_rate_pct':self.fil_supply_discount_rate+1,
            'start_discount_rate_pct':self.fil_supply_discount_rate,
            'behavior':'constant',
            'seed':1234
        }
        return fil_supply_discount_rate_process_kwargs
    
def filecoin_model_kwargs(sdm_enable_date, sdm_slope):
    def duration_master_fn(d, slope=1, clip=None):
        if d < round(365*1.5):
            return 1
        else:
            y1_slp1 = 1
            y2_slp1 = 2 - 183/365.
            x1_slp1 = round(365*1.5)
            x2_slp1 = 365*2
            m = (y2_slp1 - y1_slp1) / (x2_slp1 - x1_slp1)
            m *= slope
            # y-y1 = m*(x-x1)
            y = m*(d - x1_slp1) + y1_slp1
            if clip is not None:
                if y > clip:
                    y = clip
            return y

    def sdm_fn(date_in=None, sector_duration_days=365, sdm_slope=0.285):
        if date_in is None:
            raise ValueError('date_in must be specified')
        if date_in < sdm_enable_date:
            return 1
        else:
            return duration_master_fn(sector_duration_days, slope=sdm_slope)

    sdm_fn_kwargs = {'sdm_slope': sdm_slope}

    return {
        'sdm': sdm_fn,
        'sdm_kwargs': sdm_fn_kwargs,
    }

if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='sdm_experiments.txt')
    args = parser.parse_args()

    output_fp = args.output_fp
    print('Writing to {}'.format(output_fp))

    experiment_names = []
    
    # Generate baseline experiments
    total_daily_rb_onboard_pib_vec = [6]
    renewal_rate_vec = [.6]
    agent_power_distribution_vec = [
        [0.3, 0.7],
        [0.5, 0.5],
        [0.7, 0.3],
    ]
    cc_split_vec = [0.7, 0.8, 0.9]  # the % of total CC agents which are risk averse

    fil_supply_discount_rate_vec = [10, 20, 30]
    filplus_agent_optimism_vec = [4]
    normal_cc_agent_optimism_vec = [4]
    risk_averse_cc_agent_optimism_vec = [4]
    
    filplus_agent_discount_rate_yr_pct_vec = [25, 50]
    normal_cc_agent_discount_rate_multiplier_vec = [1, 2]
    risk_averse_cc_agent_discount_rate_multiplier_vec = [2, 3, 4]

    sdm_enable_date = date(2023, 10, 15) # ~6 months after the start of the simulation
    sdm_slope_vec = [1.0, 0.285]

    for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for agent_power_distribution in agent_power_distribution_vec:
                for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                    for filplus_agent_optimism in filplus_agent_optimism_vec:
                        for base_agent_discount_rate_yr_pct in filplus_agent_discount_rate_yr_pct_vec:
                            filplus_agent_discount_rate = base_agent_discount_rate_yr_pct
                            for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                                for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                    normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * base_agent_discount_rate_yr_pct
                                    for sdm_slope in sdm_slope_vec:
                                        # baseline experiment
                                        name = 'SDMBaseline_%0.03f,FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                            (
                                                sdm_slope,
                                                filplus_agent_optimism, filplus_agent_discount_rate, agent_power_distribution[0],
                                                normal_cc_agent_optimism, normal_cc_agent_discount_rate, 
                                                total_daily_rb_onboard_pib, renewal_rate, fil_supply_discount_rate,
                                            )
                                        experiment_names.append(name)

                                        # test experiments
                                        for cc_split in cc_split_vec:
                                            for risk_averse_cc_agent_optimism in risk_averse_cc_agent_optimism_vec:
                                                for risk_averse_cc_agent_discount_rate_multiplier in risk_averse_cc_agent_discount_rate_multiplier_vec:
                                                    risk_averse_cc_agent_discount_rate = risk_averse_cc_agent_discount_rate_multiplier * base_agent_discount_rate_yr_pct
                                                    name = 'SDMExperiment_%0.03f,FILP_%d,%d,%0.02f,NormalCC_%d,%d,RACC_%d,%d,CCSplit_%0.02f,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                                    (
                                                        sdm_slope,
                                                        filplus_agent_optimism, filplus_agent_discount_rate, agent_power_distribution[0],
                                                        normal_cc_agent_optimism, normal_cc_agent_discount_rate, 
                                                        risk_averse_cc_agent_optimism, risk_averse_cc_agent_discount_rate,
                                                        cc_split, 
                                                        total_daily_rb_onboard_pib, 
                                                        renewal_rate, 
                                                        fil_supply_discount_rate,
                                                    )
                                                    experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)
