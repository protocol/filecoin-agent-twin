from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.dca_agent import DCAAgent, DCAAgentWithTerminate
import agentfil.constants as C

import argparse

class ExpDCAAgentsTerminate(ExperimentCfg):
    def __init__(self, num_agents, 
                 agent_power_distribution,
                 subpopulation_terminate_pct,
                 agent_max_sealing_throughput,
                 max_daily_rb_onboard_pib, 
                 renewal_rate, 
                 fil_plus_rate, 
                 sector_duration,
                 fil_supply_discount_rate,
                 terminate_date):
        
        self.num_agents = num_agents
        assert self.num_agents == 3, "Only support 3 agent subpopulations currently"
        self.agent_max_sealing_throughput = agent_max_sealing_throughput
        self.agent_power_distribution = agent_power_distribution
        self.subpopulation_terminate_pct = subpopulation_terminate_pct
        self.terminate_date = terminate_date

        # agent related configuration
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.sector_duration = sector_duration
        
        # external environment configuration
        self.fil_supply_discount_rate = fil_supply_discount_rate

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [DCAAgent, DCAAgentWithTerminate] * self.num_agents
        agent_power_distribution = []
        
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            if ii == 0:
                agent_fil_plus_rate = 1.0  # this is configuring the FIL+ agents
            elif ii == 1:
                agent_fil_plus_rate = 0.0  # this is the CC agent
            elif ii == 2:
                agent_fil_plus_rate = self.fil_plus_rate  # this is the mixed agent
            
            nonterminate_agent_power = self.agent_power_distribution[ii] * (1-self.subpopulation_terminate_pct)
            agent_onboard_pib = self.max_daily_rb_onboard_pib * nonterminate_agent_power
            agent_kwargs = {
                'max_daily_rb_onboard_pib': agent_onboard_pib,
                'max_sealing_throughput': self.agent_max_sealing_throughput,
                'renewal_rate': self.renewal_rate,
                'fil_plus_rate': agent_fil_plus_rate,
                'sector_duration': self.sector_duration,
            }
            agent_kwargs_vec.append(agent_kwargs)
            agent_power_distribution.append(nonterminate_agent_power)

            terminate_agent_power = self.agent_power_distribution[ii] * self.subpopulation_terminate_pct
            agent_onboard_pib = self.max_daily_rb_onboard_pib * terminate_agent_power
            agent_with_terminate_kwargs = {
                'max_daily_rb_onboard_pib': agent_onboard_pib,
                'max_sealing_throughput': self.agent_max_sealing_throughput,
                'renewal_rate': self.renewal_rate,
                'fil_plus_rate': self.fil_plus_rate,
                'sector_duration': self.sector_duration,
                'terminate_date': self.terminate_date,
            }
            agent_kwargs_vec.append(agent_with_terminate_kwargs)
            agent_power_distribution.append(terminate_agent_power)

        return agent_types, agent_kwargs_vec, agent_power_distribution
    
    def get_fil_supply_discount_rate_process_cfg(self):
        fil_supply_discount_rate_process_kwargs = {
            'min_discount_rate_pct':self.fil_supply_discount_rate-1, 
            'max_discount_rate_pct':self.fil_supply_discount_rate+1,
            'start_discount_rate_pct':self.fil_supply_discount_rate,
            'behavior':'constant',
            'seed':1234
        }
        
        return fil_supply_discount_rate_process_kwargs
    
def generate_terminate_experiments(output_fp):
    print('Writing to {}'.format(output_fp))

    experiment_names = []

    population_power_breakdown = [
        [0.33, 0.33, 0.34],
        [0.495, 0.495, 0.01],
        [0.695, 0.295, 0.01],
    ]
    subpopulation_terminate_pcts = [0.0, 0.3, 0.5, 0.7]

    total_onboard_rbp = 6  # across all agents in the simulation
    renewal_rate = 0.6     # for agents which decide to stay on the network
    fil_plus_rate = 0.8    # for the mixed agents which decide to stay on the network
    fil_supply_discount_rate_vec = [10, 20, 30]

    for population_power in population_power_breakdown:
        agent_power_distribution = population_power
        for subpopulation_terminate_pct in subpopulation_terminate_pcts:
            for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                name = 'Terminate_%0.02f-FP_%0.02f-CC_%0.02f-MX_%0.02f-MaxRBP_%0.02f-RR_%0.02f-FPR_%0.02f-DR_%d' % \
                    (subpopulation_terminate_pct, 
                    agent_power_distribution[0], agent_power_distribution[1], agent_power_distribution[2],
                    total_onboard_rbp, renewal_rate, fil_plus_rate, fil_supply_discount_rate)
                experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)


if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='dca_terminate.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_terminate_experiments(output_fp)