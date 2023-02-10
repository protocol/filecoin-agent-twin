from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.greedy_agent import GreedyAgent

class ExpGreedyAgentsConstantOptimismTemplate(ExperimentCfg):
    def __init__(self, num_agents, agent_optimism):
        self.num_agents = num_agents
        self.agent_optimism = agent_optimism

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [GreedyAgent] * self.num_agents
        
        random_seed_base = 1000
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            agent_kwargs = {
                'agent_optimism': self.agent_optimism,
                'random_seed': ii + random_seed_base
            }
            agent_kwargs_vec.append(agent_kwargs)

        # uniform power distribution
        agent_power_distribution = np.ones(self.num_agents) / self.num_agents

        return agent_types, agent_kwargs_vec, agent_power_distribution
