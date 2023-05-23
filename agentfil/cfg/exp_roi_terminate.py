from typing import Dict, Tuple, List

import numpy as np

from agentfil.cfg.experiment_cfg import ExperimentCfg
from agentfil.agents.dca_agent import DCAAgentWithTerminate
from agentfil.agents.roi_agent import ROIAgentDynamicOnboard
import agentfil.constants as C

import argparse

class ExpROIAdaptDCATerminate(ExperimentCfg):
    def __init__(self, num_agents, 
                 agent_power_distribution,
                 subpopulation_terminate_pct,
                 
                 max_sealing_throughput,
                 min_daily_rb_onboard_pib,
                 max_daily_rb_onboard_pib,
                 min_renewal_rate,   
                 max_renewal_rate,
                 fil_plus_rate,
                 min_roi,
                 max_roi,
                 roi_agent_optimism,

                 sector_duration,
                 fil_supply_discount_rate,
                 terminate_date):

        self.num_agents = num_agents
        assert self.num_agents == 3, "Only support 3 agent subpopulations currently"
        self.agent_power_distribution = agent_power_distribution
        self.subpopulation_terminate_pct = subpopulation_terminate_pct
        self.terminate_date = terminate_date

        # pertinent to the ROI agent
        self.roi_agent_min_roi = min_roi
        self.roi_agent_max_roi = max_roi
        self.roi_agent_optimism = roi_agent_optimism
        self.roi_agent_min_daily_rb_onboard_pib = min_daily_rb_onboard_pib
        self.roi_agent_max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.roi_agent_min_renewal_rate = min_renewal_rate
        self.roi_agent_max_renewal_rate = max_renewal_rate

        # pertinent to terminate agent only
        self.terminate_agent_max_daily_rb_onboard_pib = min_daily_rb_onboard_pib
        self.terminate_agent_renewal_rate = min_renewal_rate
        self.terminate_agent_sector_duration = sector_duration
        
        # common to both agents
        self.fil_plus_rate = fil_plus_rate
        self.agent_max_sealing_throughput = max_sealing_throughput
        
        # external environment configuration
        self.fil_supply_discount_rate = fil_supply_discount_rate

    def get_agent_cfg(self) -> Tuple[List, List, List]:
        agent_types = [ROIAgentDynamicOnboard, DCAAgentWithTerminate] * self.num_agents
        agent_power_distribution = []
        
        agent_kwargs_vec = []
        for ii in range(self.num_agents):
            if ii == 0:
                agent_fil_plus_rate = 1.0  # this is configuring the FIL+ agents
            elif ii == 1:
                agent_fil_plus_rate = 0.0  # this is the CC agent
            elif ii == 2:
                agent_fil_plus_rate = self.fil_plus_rate  # this is the mixed agent
            
            roi_agent_power = self.agent_power_distribution[ii] * (1-self.subpopulation_terminate_pct)
            agent_kwargs = {
                'max_sealing_throughput': self.agent_max_sealing_throughput,
                'min_daily_rb_onboard_pib': self.roi_agent_min_daily_rb_onboard_pib * roi_agent_power,
                'max_daily_rb_onboard_pib': self.roi_agent_max_daily_rb_onboard_pib * roi_agent_power,
                'min_renewal_rate': self.roi_agent_min_renewal_rate,
                'max_renewal_rate': self.roi_agent_max_renewal_rate,
                'fil_plus_rate': agent_fil_plus_rate,
                'min_roi': self.roi_agent_min_roi,
                'max_roi': self.roi_agent_max_roi,
                'agent_optimism': self.roi_agent_optimism
            }
            agent_kwargs_vec.append(agent_kwargs)
            agent_power_distribution.append(roi_agent_power)

            terminate_agent_power = self.agent_power_distribution[ii] * self.subpopulation_terminate_pct
            agent_onboard_pib = self.terminate_agent_max_daily_rb_onboard_pib * terminate_agent_power
            agent_with_terminate_kwargs = {
                'max_daily_rb_onboard_pib': agent_onboard_pib,
                'max_sealing_throughput': self.agent_max_sealing_throughput,
                'renewal_rate': self.terminate_agent_renewal_rate,
                'fil_plus_rate': self.fil_plus_rate,
                'sector_duration': self.terminate_agent_sector_duration,
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
    
    def get_rewards_per_sector_process_cfg(self) -> Dict:
        return {
            'update_every_days':30,
            'linear_forecast_deviation_pct': 0.3,
            'verbose': False,
            'keep_previous_predictions': False,
            'keep_power_predictions': False,
        }
    
def generate_terminate_experiments(output_fp):
    print('Writing to {}'.format(output_fp))
    experiment_names = []

    population_power_breakdown = [
        [0.33, 0.33, 0.34],
        [0.495, 0.495, 0.01],
        [0.695, 0.295, 0.01],
        [0.295, 0.695, 0.01],
    ]
    subpopulation_terminate_pcts = [0.0, 0.3, 0.5, 0.7]

    total_min_onboard_rbp = 0
    total_max_onboard_rbp_vec = [3,6,15]
    min_rr = 0.0
    max_rr_vec = [0.4, 0.8]
    min_roi_vec = [0.1, 0.3]
    max_roi_vec = [0.8, 1.0]
    roi_agent_optimism_vec = [2,4]
    fil_plus_rate = 0.8    # for the mixed agents which decide to stay on the network
    fil_supply_discount_rate = 10  # a noop when using ROI agents

    for population_power in population_power_breakdown:
        agent_power_distribution = population_power
        for subpopulation_terminate_pct in subpopulation_terminate_pcts:
            for total_max_onboard_rbp in total_max_onboard_rbp_vec:
                for max_rr in max_rr_vec:
                    for min_roi in min_roi_vec:
                        for max_roi in max_roi_vec:
                            for roi_agent_optimism in roi_agent_optimism_vec:
                                name = 'ROI_%d_%0.2f_%0.02f-Terminate_%0.02f-FP_%0.02f-CC_%0.02f-MX_%0.02f-MinRBP_%0.02f-MaxRBP_%0.02f-MinRR_%0.02f-MaxRR_%0.02f-FPR_%0.02f-DR_%d' % \
                                    (roi_agent_optimism, min_roi, max_roi, subpopulation_terminate_pct, 
                                        agent_power_distribution[0], agent_power_distribution[1], agent_power_distribution[2],
                                        total_min_onboard_rbp, total_max_onboard_rbp, min_rr, max_rr,
                                        fil_plus_rate, fil_supply_discount_rate)
                                experiment_names.append(name)

    with open(output_fp, 'w')  as f:
        for name in experiment_names:
            f.write('%s\n' % name)


if __name__ == '__main__':
    # Generate configurations for the SDM experiments and write them to a config file
    # that can be used by the experiment runner
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_fp', type=str, default='roi_terminate.txt')
    
    args = parser.parse_args()
    output_fp = args.output_fp

    generate_terminate_experiments(output_fp)