from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.farsighted_agent import FarSightedAgent
from agentfil.filecoin_model import distribute_agent_power_geometric_series

class ExpFarsightedAgentsConstantOptimismUniformPowerDistribution(ExperimentCfg):
    def __init__(self, num_agents, agent_optimism, 
                 far_sightedness_days=90, reestimate_every_days=90):
        self.num_agents = num_agents
        self.agent_optimism = agent_optimism
        self.far_sightedness_days = far_sightedness_days
        self.reestimate_every_days = reestimate_every_days

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [FarSightedAgent] * self.num_agents
        
        random_seed_base = 1000
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            agent_kwargs = {
                'random_seed': ii + random_seed_base,
                'agent_optimism': self.agent_optimism,
                'far_sightedness_days': self.far_sightedness_days,
                'reestimate_every_days': self.reestimate_every_days
            }
            agent_kwargs_vec.append(agent_kwargs)

        # uniform power distribution
        agent_power_distribution = np.ones(self.num_agents) / self.num_agents

        return agent_types, agent_kwargs_vec, agent_power_distribution

class ExpFarsightedAgentsConstantOptimismGeometricPowerDistribution(ExperimentCfg):
    def __init__(self, num_agents, agent_optimism, 
                 max_agent_power=0.2,
                 far_sightedness_days=90, 
                 reestimate_every_days=90):
        self.num_agents = num_agents
        self.agent_optimism = agent_optimism
        self.max_agent_power = max_agent_power
        self.far_sightedness_days = far_sightedness_days
        self.reestimate_every_days = reestimate_every_days

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [FarSightedAgent] * self.num_agents
        
        random_seed_base = 1000
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            agent_kwargs = {
                'random_seed': ii + random_seed_base,
                'agent_optimism': self.agent_optimism,
                'far_sightedness_days': self.far_sightedness_days,
                'reestimate_every_days': self.reestimate_every_days
            }
            agent_kwargs_vec.append(agent_kwargs)

        # uniform power distribution
        agent_power_distribution = distribute_agent_power_geometric_series(self.num_agents, a=self.max_agent_power)

        return agent_types, agent_kwargs_vec, agent_power_distribution

class ExpFarsightedAgentsProportionalOptimismGeometricPowerDistribution(ExperimentCfg):
    def __init__(self, num_agents, 
                 max_agent_power=0.2, 
                 far_sightedness_days=90, 
                 reestimate_every_days=90,
                 min_optimism=2, 
                 max_optimism=4):
        self.num_agents = num_agents
        self.max_agent_power = max_agent_power
        self.far_sightedness_days = far_sightedness_days
        self.reestimate_every_days = reestimate_every_days
        self.min_optimism = min_optimism
        self.max_optimism = max_optimism

        self.validate()

    def validate(self):
        if self.min_optimism >= self.max_optimism:
            raise ValueError('min_optimism must be less than max_optimism')

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        # geometric power distribution, this list is sorted in descending order by definition
        agent_power_distribution = distribute_agent_power_geometric_series(self.num_agents, a=self.max_agent_power)
        agent_types = [FarSightedAgent] * self.num_agents
        
        optimism_choices = np.arange(self.min_optimism, self.max_optimism+1, 1)
        bin_edges = np.histogram_bin_edges(optimism_choices, bins=len(optimism_choices)-1)
        optimism_idxs = np.digitize(agent_power_distribution, bin_edges)-1  # change 1-indexing to 0-indexing

        random_seed_base = 1000
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            agent_kwargs = {
                'random_seed': ii + random_seed_base,
                'agent_optimism': int(optimism_choices[optimism_idxs[ii]]),
                'far_sightedness_days': self.far_sightedness_days,
                'reestimate_every_days': self.reestimate_every_days
            }
            agent_kwargs_vec.append(agent_kwargs)

        return agent_types, agent_kwargs_vec, agent_power_distribution