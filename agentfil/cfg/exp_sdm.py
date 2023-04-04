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
    def __init__(self, 
                 max_sealing_throughput,
                 max_daily_onboard_rb_pib, renewal_rate,
                 agent_power_distribution,
                 fil_supply_discount_rate,
                 filplus_agent_optimism, filplus_agent_discount_rate_yr_pct,
                 cc_agent_optimism, cc_agent_discount_rate_yr_pct):
        
        self.agent_types = [NPVAgent, NPVAgent]
        self.num_agents = len(self.agent_types)
        self.agent_kwargs = [
            {
                'max_sealing_throughput': max_sealing_throughput,
                'max_daily_rb_onboard_pib': max_daily_onboard_rb_pib,
                'renewal_rate': 0,   # renewals are not allowed for deals, so the FIL+ agent doesn't renew any power
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