from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.dca_agent import DCAAgent

class ExpDCAAgentsConstantDiscountRate(ExperimentCfg):
    def __init__(self, num_agents, 
                 agent_power_distribution,
                 agent_max_sealing_throughput,
                 max_daily_rb_onboard_pib, 
                 renewal_rate, 
                 fil_plus_rate, 
                 sector_duration,
                 fil_supply_discount_rate):
        
        self.num_agents = num_agents
        self.agent_max_sealing_throughput = agent_max_sealing_throughput
        self.agent_power_distribution = agent_power_distribution

        # agent related configuration
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.sector_duration = sector_duration
        
        # external environment configuration
        self.fil_supply_discount_rate = fil_supply_discount_rate

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [DCAAgent] * self.num_agents
        
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            agent_kwargs = {
                'max_daily_rb_onboard_pib': self.max_daily_rb_onboard_pib,
                'max_sealing_throughput': self.agent_max_sealing_throughput[ii],
                'renewal_rate': self.renewal_rate,
                'fil_plus_rate': self.fil_plus_rate,
                'sector_duration': self.sector_duration,
            }
            agent_kwargs_vec.append(agent_kwargs)

        return agent_types, agent_kwargs_vec, self.agent_power_distribution
    
    def get_fil_supply_discount_rate_process_cfg(self):
        fil_supply_discount_rate_process_kwargs = {
            'min_discount_rate_pct':self.fil_supply_discount_rate-1, 
            'max_discount_rate_pct':self.fil_supply_discount_rate+1,
            'start_discount_rate_pct':self.fil_supply_discount_rate,
            'behavior':'constant',
            'seed':1234
        }
        
        return fil_supply_discount_rate_process_kwargs

class ExpDCAAgentsLinearAdaptiveDiscountRate(ExperimentCfg):
    def __init__(self, num_agents, 
                 agent_power_distribution,
                 agent_max_sealing_throughput,
                 max_daily_rb_onboard_pib, 
                 renewal_rate, 
                 fil_plus_rate, 
                 sector_duration,
                 min_discount_rate_pct,
                 max_discount_rate_pct,
                 start_discount_rate_pct,):
        
        self.num_agents = num_agents
        self.agent_max_sealing_throughput = agent_max_sealing_throughput
        self.agent_power_distribution = agent_power_distribution

        # agent related configuration
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.sector_duration = sector_duration

        # external environment configuration
        self.min_discount_rate_pct = min_discount_rate_pct
        self.max_discount_rate_pct = max_discount_rate_pct
        self.start_discount_rate_pct = start_discount_rate_pct

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [DCAAgent] * self.num_agents
        
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            agent_kwargs = {
                'max_daily_rb_onboard_pib': self.max_daily_rb_onboard_pib,
                'max_sealing_throughput': self.agent_max_sealing_throughput[ii],
                'renewal_rate': self.renewal_rate,
                'fil_plus_rate': self.fil_plus_rate,
                'sector_duration': self.sector_duration,
            }
            agent_kwargs_vec.append(agent_kwargs)

        return agent_types, agent_kwargs_vec, self.agent_power_distribution
    
    def get_fil_supply_discount_rate_process_cfg(self):
        fil_supply_discount_rate_process_kwargs = {
            'min_discount_rate_pct':self.min_discount_rate_pct,
            'max_discount_rate_pct':self.max_discount_rate_pct,
            'start_discount_rate_pct':self.start_discount_rate_pct,
            'behavior':'linear-adaptive',
            'seed':1234
        }
        
        return fil_supply_discount_rate_process_kwargs
