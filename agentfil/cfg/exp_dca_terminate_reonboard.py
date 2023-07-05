from typing import Dict, Tuple, List

from datetime import timedelta, date
import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.dca_agent import DCAAgentTerminate, DCAAgentTwoModes
import agentfil.constants as C

import argparse

class ExpDCAAgentsTerminateReonboard(ExperimentCfg):
    def __init__(self, 
                 subpopulation_pct,
                 agent_max_sealing_throughput,
                 before_terminate_agent_behavior,
                 after_terminate_stay_agent_behavior,
                 sector_duration,
                 fil_supply_discount_rate,
                 terminate_date):
        
        self.agent_max_sealing_throughput = agent_max_sealing_throughput
        self.agent_power_distribution = [1.0]
        self.subpopulation_pct = subpopulation_pct
        self.terminate_date = terminate_date

        self.before_terminate_agent_behavior = before_terminate_agent_behavior
        self.after_terminate_stay_agent_behavior = after_terminate_stay_agent_behavior

        # agent related configuration
        self.sector_duration = sector_duration
        
        # external environment configuration
        self.fil_supply_discount_rate = fil_supply_discount_rate

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [DCAAgentTwoModes, DCAAgentTerminate]
        agent_power_distribution = []
        
        agent_kwargs_vec = []
                    
        nonterminate_agent_power = self.subpopulation_pct
        agent_kwargs = {
            'max_sealing_throughput': self.agent_max_sealing_throughput,
            'mode1_behavior': self.before_terminate_agent_behavior,
            'mode2_date': self.terminate_date + timedelta(days=1),
            'mode2_behavior': self.after_terminate_stay_agent_behavior,
            'sector_duration': self.sector_duration,
        }
        agent_kwargs_vec.append(agent_kwargs)
        agent_power_distribution.append(nonterminate_agent_power)

        terminate_agent_power = (1-self.subpopulation_pct)
        agent_with_terminate_kwargs = {
            'max_daily_rb_onboard_pib': self.before_terminate_agent_behavior['rbp'],
            'max_sealing_throughput': self.agent_max_sealing_throughput,
            'renewal_rate': self.before_terminate_agent_behavior['rr'],
            'fil_plus_rate': self.before_terminate_agent_behavior['fpr'],
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
    subpopulation_pcts = [0.3, 0.7]  # v = % of agents that stay, 1-v = % of agents that terminate
                                 # in the first case, 30% of agents stay and 70% of agents terminate
                                 # in the second case, 70% of agents stay and 30% of agents terminate
    terminate_date = date(2023, 11, 1)

    agent_behavior_before_terminate = {
        'rbp':6,  # across all agents in the simulation
        'rr':0.6,     # for agents which decide to stay on the network
        'fpr':0.8    # for the mixed agents which decide to stay on the network
    }

    stay_agent_behavior_after_terminate_vec = [
        {'rbp':6, 'rr':0.6, 'fpr':0.8},
        {'rbp':20, 'rr':0.8, 'fpr':0.8},
        {'rbp':50, 'rr':0.8, 'fpr':0.8},
        {'rbp':100, 'rr':0.9, 'fpr':0.8}
    ]
    max_possible_rbp = 100

    sector_duration = 360

    for subpopulation_pct in subpopulation_pcts:
        for stay_agent_behavior in stay_agent_behavior_after_terminate_vec:
            name = 'TerminateReonboard_%0.02f_MaxRBP_%0.02f_%0.02f-RR_%0.02f_%0.02f-FPR_%0.02f_%0.02f' % \
            (subpopulation_pct, 
             agent_behavior_before_terminate['rbp'], stay_agent_behavior['rbp'], 
             agent_behavior_before_terminate['rr'], stay_agent_behavior['rr'], 
             agent_behavior_before_terminate['fpr'], stay_agent_behavior['fpr'])
            experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)


if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='dca_terminate_reonboard.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_terminate_experiments(output_fp)