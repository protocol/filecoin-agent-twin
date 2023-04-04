"""
We have two experiments here
Baseline - FIL+ and CC NPV agents, with a proportional power distribution.
Experiment - FIL+, CC, and Risk Averse CC NPV agents, with a proportional power distribution.
  As we sweep the percentage of risk-averse NPV agents, what happens to the network KPI?
"""

from typing import Dict, Tuple, List

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
                 max_daily_onboard_rb_pib, renewal_rate,
                 agent_power_distribution,
                 fil_supply_discount_rate,
                 filplus_agent_optimism, filplus_agent_discount_rate_yr_pct,
                 cc_agent_optimism, cc_agent_discount_rate_yr_pct):
        assert sum(agent_power_distribution) == 1.0, "agent_power_distribution must sum to 1.0"

        self.agent_types = [NPVAgent, NPVAgent]
        self.num_agents = len(self.agent_types)
        self.agent_kwargs = [
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': max_daily_onboard_rb_pib,
                'renewal_rate': renewal_rate,   
                'fil_plus_rate': 1,  # 100% FIL+ agent
                'agent_optimism': filplus_agent_optimism,
                'agent_discount_rate_yr_pct': filplus_agent_discount_rate_yr_pct,
            },
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': max_daily_onboard_rb_pib,
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
    
class SDMControlExperiment(ExperimentCfg):
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
                 max_daily_onboard_rb_pib, renewal_rate,
                 agent_power_distribution,
                 fil_supply_discount_rate,
                 filplus_agent_optimism, filplus_agent_discount_rate_yr_pct,
                 normal_cc_agent_optimism, normal_cc_agent_discount_rate_yr_pct,
                 riskaverse_cc_agent_optimism, riskaverse_cc_agent_discount_rate_yr_pct):
        assert sum(agent_power_distribution) == 1.0, "agent_power_distribution must sum to 1.0"

        self.agent_types = [NPVAgent, NPVAgent, NPVAgent]
        self.num_agents = len(self.agent_types)
        self.agent_kwargs = [
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': max_daily_onboard_rb_pib,
                'renewal_rate': renewal_rate,   
                'fil_plus_rate': 1,  # 100% FIL+ agent
                'agent_optimism': filplus_agent_optimism,
                'agent_discount_rate_yr_pct': filplus_agent_discount_rate_yr_pct,
            },
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': max_daily_onboard_rb_pib,
                'renewal_rate': renewal_rate,
                'fil_plus_rate': 0,  # a CC agent
                'agent_optimism': normal_cc_agent_optimism,
                'agent_discount_rate_yr_pct': normal_cc_agent_discount_rate_yr_pct,
            },
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': max_daily_onboard_rb_pib,
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