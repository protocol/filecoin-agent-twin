from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.dca_agent import DCAAgent
import agentfil.constants as C

import argparse

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
    
class ExpDCAAgentsPowerScaledConstantDiscountRate(ExperimentCfg):
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
                'max_daily_rb_onboard_pib': self.max_daily_rb_onboard_pib * self.agent_power_distribution[ii],
                'max_sealing_throughput': self.agent_max_sealing_throughput,
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


def generate_dca_experiments(output_fp):
    print('Writing to {}'.format(output_fp))

    experiment_names = []

    agent_power_distribution_vec = [
        np.asarray([1,1,1,1,1]),  # each agent has 20% of the total power
        np.asarray([2,2,2,1,1]),  # first 3 agents have 25% of total power, last 2 agents have 12.5% of total power
        np.asarray([4,1,1,1,1]),  # first agent has 50% of total power, last 4 agents have 12.5% of total power
        np.asarray([5,4,3,2,1]),  # power distribution = [5/15, 4/15, 3/15, 2/15, 1/15] * 100%
    ]

    max_total_onboard_pib_vec = [6,50]
    renewal_rate_vec = [.6]
    fil_plus_rate_vec = [.6]
    sector_duration_vec = [360]
    fil_supply_discount_rate_vec = [10, 20, 30]

    for agent_power_distribution in agent_power_distribution_vec:
        for max_total_onboard_pib in max_total_onboard_pib_vec:
            for renewal_rate in renewal_rate_vec:
                for fil_plus_rate in fil_plus_rate_vec:
                    for sector_duration in sector_duration_vec:
                        for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                            agent_power_distribution_str = ','.join([str(x) for x in agent_power_distribution])
                            name = 'DCAPowerConcentration=%s-ConstFilSupplyDiscountRate=%d-MaxDailyOnboard=%0.02f-RenewalRate=%0.02f-FilPlusRate=%0.02f-SectorDuration=%d' % \
                                (agent_power_distribution_str,fil_supply_discount_rate, max_total_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                            experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)


if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='power_concentration.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_dca_experiments(output_fp)