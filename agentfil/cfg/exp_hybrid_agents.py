"""
Here, we use DCA agents to account for network "baseline." We then add agents on top of that with various
configurations to see how they affect the network & agent reward structure.
"""

from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.dca_agent import DCAAgent

class ExpHybridConstantDiscountRate(ExperimentCfg):
    def __init__(self, 
                 agent_types,
                 agent_kwargs,
                 agent_power_distribution,
                 fil_supply_discount_rate):
        
        self.agent_types = agent_types
        self.num_agents = len(agent_types)
        self.agent_kwargs = agent_kwargs
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