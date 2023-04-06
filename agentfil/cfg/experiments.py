import numpy as np

from datetime import date

from .. import constants as C
from ..agents import dca_agent, basic_rational_agent, roi_agent, npv_agent
from . import exp_sdm

"""
Add experiments to this file so that they can be run from the command line.
"""

name2experiment = {}

"""
Experiments related to: "How does the discount rate affect the agent rewards"
This is the most basic manifestation, since we use DCA agents which onboard a constant
amount of power, renew a constant amount of power, and have a constant FIL+ percentage.
The discount rate is also static throughout the simulation.
"""
num_agents_vec = [1]
max_daily_rb_onboard_pib_vec = [5, 10, 25]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3, 360*5]
fil_supply_discount_rate_vec = [20, 25, 30, 35, 40, 45, 50]
max_sealing_throughput = [C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB]

for num_agents in num_agents_vec:
    for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for fil_plus_rate in fil_plus_rate_vec:
                for sector_duration in sector_duration_vec:
                    for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                        name = 'DCA-ConstFilSupplyDiscountRate=%d-MaxDailyOnboard=%0.02f-RenewalRate=%0.02f-FilPlusRate=%0.02f-SectorDuration=%d' % \
                            (fil_supply_discount_rate, max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                        name2experiment[name] = {
                            'module_name': 'agentfil.cfg.exp_dca_agents',
                            'instantiator': 'ExpDCAAgentsConstantDiscountRate',
                            'instantiator_kwargs': {
                                'num_agents': num_agents,
                                'agent_max_sealing_throughput': max_sealing_throughput,
                                'agent_power_distribution': [1],
                                'max_daily_rb_onboard_pib': max_daily_rb_onboard_pib,
                                'renewal_rate': renewal_rate,
                                'fil_plus_rate': fil_plus_rate,
                                'sector_duration': sector_duration,
                                'fil_supply_discount_rate': fil_supply_discount_rate
                            },
                            'filecoin_model_kwargs': {},
                        }

"""
Experiments related to: "How does the discount rate affect the agent rewards"
This is the most basic manifestation, since we use DCA agents which onboard a constant
amount of power, renew a constant amount of power, and have a constant FIL+ percentage.
Here, the discount rate changes over the course of the simulation according to a "linear-adaptive"
mechanism.  

TODO: describe how the linear-adaptive mechanism works
"""
num_agents_vec = [1]
max_daily_rb_onboard_pib_vec = [5, 10, 25]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3, 360*5]
min_discount_rate_pct_vec = [10, 15, 20]
max_discount_rate_pct_vec = [50, 75, 100]
start_discount_rate_pct = 0 # a noop, will be overwritten
max_sealing_throughput = [C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB]

for num_agents in num_agents_vec:
    for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for fil_plus_rate in fil_plus_rate_vec:
                for sector_duration in sector_duration_vec:
                    for min_discount_rate_pct in min_discount_rate_pct_vec:
                        for max_discount_rate_pct in max_discount_rate_pct_vec:
                            name = 'DCA-LinearAdaptiveFilSupplyDiscountRate=%d,%d-MaxDailyOnboard=%0.02f-RenewalRate=%0.02f-FilPlusRate=%0.02f-SectorDuration=%d' % \
                                (min_discount_rate_pct, max_discount_rate_pct, max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                            name2experiment[name] = {
                                'module_name': 'agentfil.cfg.exp_dca_agents',
                                'instantiator': 'ExpDCAAgentsLinearAdaptiveDiscountRate',
                                'instantiator_kwargs': {
                                    'num_agents': num_agents,
                                    'agent_max_sealing_throughput': max_sealing_throughput,
                                    'agent_power_distribution': [1],
                                    'max_daily_rb_onboard_pib': max_daily_rb_onboard_pib,
                                    'renewal_rate': renewal_rate,
                                    'fil_plus_rate': fil_plus_rate,
                                    'sector_duration': sector_duration,
                                    'min_discount_rate_pct': min_discount_rate_pct,
                                    'max_discount_rate_pct': max_discount_rate_pct,
                                    'start_discount_rate_pct': start_discount_rate_pct
                                },
                                'filecoin_model_kwargs': {},
                            }

"""
Experiments which increase the complexity slighty. Here, we keep the DCA agent as is, but 
explore the question of the sensitivity of the agent rewards to the discount rate, when the
maximum sealing throughput of the agents is varied.

The maximum sealing throughput will translate to the share of total network QAP that each
agent receives, so we expect to see that larger agents are better able whether higher 
discount rates than smaller agents.
"""
num_agents_vec = [5]
agent_power_distribution_vec = [
    np.asarray([1,1,1,1,1]),  # each agent has 20% of the total power
    np.asarray([2,2,2,1,1]),  # first 3 agents have 25% of total power, last 2 agents have 12.5% of total power
    np.asarray([4,1,1,1,1]),  # first agent has 50% of total power, last 4 agents have 12.5% of total power
    np.asarray([5,4,3,2,1]),  # power distribution = [5/15, 4/15, 3/15, 2/15, 1/15] * 100%
]

max_daily_rb_onboard_pib_vec = [5, 10, 25]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3, 360*5]
fil_supply_discount_rate_vec = [20, 30, 40, 50]

for num_agents in num_agents_vec:
    for agent_power_distribution in agent_power_distribution_vec:
        # normalize to sum to one
        agent_power_distribution_normalized = agent_power_distribution/np.sum(agent_power_distribution)
        # setup the max sealing throughput to be proportional to agent power, capped by the default max
        max_sealing_throughput_scaled = np.sum(agent_power_distribution)/max(agent_power_distribution) * C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB
        max_sealing_throughput = agent_power_distribution_normalized * max_sealing_throughput_scaled

        for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
            for renewal_rate in renewal_rate_vec:
                for fil_plus_rate in fil_plus_rate_vec:
                    for sector_duration in sector_duration_vec:
                        for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                            agent_power_distribution_str = ','.join([str(x) for x in agent_power_distribution])
                            name = 'DCA=%s-ConstFilSupplyDiscountRate=%d-MaxDailyOnboard=%0.02f-RenewalRate=%0.02f-FilPlusRate=%0.02f-SectorDuration=%d' % \
                                (agent_power_distribution_str,fil_supply_discount_rate, max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                            name2experiment[name] = {
                                'module_name': 'agentfil.cfg.exp_dca_agents',
                                'instantiator': 'ExpDCAAgentsConstantDiscountRate',
                                'instantiator_kwargs': {
                                    'num_agents': num_agents,
                                    'agent_power_distribution': agent_power_distribution_normalized,
                                    'agent_max_sealing_throughput': max_sealing_throughput,
                                    'max_daily_rb_onboard_pib': max_daily_rb_onboard_pib,
                                    'renewal_rate': renewal_rate,
                                    'fil_plus_rate': fil_plus_rate,
                                    'sector_duration': sector_duration,
                                    'fil_supply_discount_rate': fil_supply_discount_rate
                                },
                                'filecoin_model_kwargs': {},
                            }



"""
Experiments that begin to add complexity to the baseline. Lets term these "Hybrid" experiments.  Here
a defined % of the network power is assigned to DCA agents, which provide a "steady-state" behavior. 
The remaining % of the network power is assigned to agents which act rationally in different ways.
We seek to understand how the proportion of steady-state to dynamic agents affects:
  1) Rewards for both types of agents
  2) The overall network econometrics
"""
max_daily_rb_onboard_pib_vec = [5, 10, 25]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3]
steady_state_total_network_power_vec = [.80, .85, .90, .95]
rational_agents_scaling_factor_vec = [.25, .50, .75, 1.0]  # scaling factor which controls the max-sealing-throughput & onboarding rate
fil_supply_discount_rate_vec = [20, 25, 30]
rational_agent_type_vec = [
    ('BasicRational', basic_rational_agent.BasicRationalAgent),
    ('ROIAgent', roi_agent.ROIAgent),
    ('NPVAgent', npv_agent.NPVAgent),
]
base_rational_agent_kwargs = {
    'renewal_rate': renewal_rate,
    'fil_plus_rate': fil_plus_rate,
}

for steady_state_total_network_power in steady_state_total_network_power_vec:
    for rational_agents_scaling_factor in rational_agents_scaling_factor_vec:
        for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
            for renewal_rate in renewal_rate_vec:
                for fil_plus_rate in fil_plus_rate_vec:
                    for sector_duration in sector_duration_vec:
                        for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                            for ii, rational_agent_type_info in enumerate(rational_agent_type_vec):
                                rational_agent_str_base = rational_agent_type_info[0]
                                rational_agent_cls = rational_agent_type_info[1]

                                agent_types = [
                                    dca_agent.DCAAgent,
                                    rational_agent_cls
                                ]

                                rational_agent_kwargs = base_rational_agent_kwargs.copy()
                                rational_agent_kwargs['max_sealing_throughput'] = C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB * rational_agents_scaling_factor
                                rational_agent_kwargs['max_daily_rb_onboard_pib'] = max_daily_rb_onboard_pib * rational_agents_scaling_factor
                                if rational_agent_str_base == 'BasicRational':
                                    rational_agent_kwargs['sector_duration'] = sector_duration
                                    rational_agent_kwargs['discount_rate_floor_pct'] = min(fil_supply_discount_rate_vec)
                                    rational_agent_str = '%s=%d,%d' % (rational_agent_str_base, sector_duration, min(fil_supply_discount_rate_vec))
                                elif rational_agent_str_base == 'ROIAgent':
                                    rational_agent_kwargs['agent_optimism'] = 4
                                    rational_agent_kwargs['roi_threshold'] = 0.1
                                    rational_agent_str = '%s=%d,%0.02f' % (rational_agent_str_base, 4, 0.1)
                                elif rational_agent_str_base == 'NPVAgent':
                                    rational_agent_kwargs['agent_optimism'] = 4
                                    rational_agent_kwargs['agent_discount_rate_yr_pct'] = 50
                                    rational_agent_str = '%s=%d,%d' % (rational_agent_str_base, 4, 50)
                                
                                # scale the rational agent by the desired scaling factor for max sealing throughput and onboarding rate
                                # don't scale the renewal rate b/c that would already be proportioned by the agent power distribution
                                # don't scale the fil_plus_rate b/c that is accounted for when scaling max-sealing-throughput and onboarding-rate
                                agent_kwargs = [
                                    # the DCA agent
                                    {
                                        'max_sealing_throughput':C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                        'max_daily_rb_onboard_pib': max_daily_rb_onboard_pib,
                                        'renewal_rate': renewal_rate,
                                        'fil_plus_rate': fil_plus_rate,
                                        'sector_duration': sector_duration,
                                    },
                                    # the rational agent
                                    rational_agent_kwargs,
                                ]
                                dca_agent_power_frac = steady_state_total_network_power
                                basic_rational_agent_power_frac = 1 - steady_state_total_network_power
                                agent_power_distribution = [dca_agent_power_frac, basic_rational_agent_power_frac]
                                
                                name = 'DCA=%0.02f,%d-%s=%0.02f,%0.02f-ConstFilSupplyDiscountRate=%d-MaxDailyOnboard=%0.02f,%0.02f-RenewalRate=%0.02f,%0.02f-FilPlusRate=%0.02f,%0.02f' % \
                                    (dca_agent_power_frac, 
                                    sector_duration,
                                    rational_agent_str,
                                    basic_rational_agent_power_frac, 
                                    rational_agents_scaling_factor,
                                    fil_supply_discount_rate, 
                                    max_daily_rb_onboard_pib, 
                                    max_daily_rb_onboard_pib, 
                                    renewal_rate, 
                                    renewal_rate, 
                                    fil_plus_rate, 
                                    fil_plus_rate, 
                                    )
                                name2experiment[name] = {
                                    'module_name': 'agentfil.cfg.exp_hybrid_agents',
                                    'instantiator': 'ExpHybridConstantDiscountRate',
                                    'instantiator_kwargs': {
                                        'agent_types': agent_types,
                                        'agent_kwargs': agent_kwargs,
                                        'agent_power_distribution': agent_power_distribution,
                                        'fil_supply_discount_rate': fil_supply_discount_rate,
                                    },
                                    'filecoin_model_kwargs': {},
                                }

"""
Experiments that relate to the evaluating the SDM policy.
These configurations are for the control experiment.
"""
total_daily_rb_onboard_pib_vec = [4, 6, 8]
renewal_rate_vec = [.4, .6, .8]
agent_power_distribution_vec = [
    [0.3, 0.7],
    [0.5, 0.5],
    [0.7, 0.3],
]
fil_supply_discount_rate_vec = [5, 10, 15, 20, 25, 30]
filplus_agent_optimism_vec = [4]
filplus_agent_discount_rate_yr_pct_vec = [25, 50]  # a representation of the agent's risk
cc_agent_optimism_vec = [4]
cc_agent_discount_rate_yr_pct_vec = [25, 50]       # a representation of the agent's risk
sdm_enable_date = date(2023, 10, 15) # ~6 months after the start of the simulation
sdm_slope_vec = [1.0, 0.285]

for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
    for renewal_rate in renewal_rate_vec:
        for agent_power_distribution in agent_power_distribution_vec:
            for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                for filplus_agent_optimism in filplus_agent_optimism_vec:
                    for filplus_agent_discount_rate in filplus_agent_discount_rate_yr_pct_vec:
                        for cc_agent_optimism in cc_agent_optimism_vec:
                            for cc_agent_discount_rate in cc_agent_discount_rate_yr_pct_vec:
                                for sdm_slope in sdm_slope_vec:

                                    filecoin_model_kwargs = exp_sdm.filecoin_model_kwargs(sdm_enable_date, sdm_slope)

                                    name = 'SDMBaseline=%0.03f,FILP=%d,%d,%0.02f,CC=%d,%d,%0.02f,Onboard=%0.02f,RR=%0.02f,DR=%d' % \
                                        (
                                            sdm_slope,
                                            filplus_agent_optimism, filplus_agent_discount_rate, agent_power_distribution[0],
                                            cc_agent_optimism, cc_agent_discount_rate, agent_power_distribution[1],
                                            total_daily_rb_onboard_pib, renewal_rate, fil_supply_discount_rate,
                                         )
                                    name2experiment[name] = {
                                        'module_name': 'agentfil.cfg.exp_sdm',
                                        'instantiator': 'SDMBaselineExperiment',
                                        'instantiator_kwargs': {
                                            'max_sealing_throughput': C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                            'total_daily_onboard_rb_pib': total_daily_rb_onboard_pib,
                                            'renewal_rate': renewal_rate,
                                            'agent_power_distribution': agent_power_distribution,
                                            'fil_supply_discount_rate': fil_supply_discount_rate,
                                            'filplus_agent_optimism': filplus_agent_optimism,
                                            'filplus_agent_discount_rate_yr_pct': filplus_agent_discount_rate,
                                            'cc_agent_optimism': cc_agent_optimism,
                                            'cc_agent_discount_rate_yr_pct': cc_agent_discount_rate,

                                        },
                                        'filecoin_model_kwargs': filecoin_model_kwargs,
                                    }

"""
Experiments that relate to the evaluating the SDM policy.
"""
total_daily_rb_onboard_pib_vec = [4, 6, 8]
renewal_rate_vec = [.4, .6, .8]
# first # represents the FIL+ agent power, second # represents the % of the remaining that is 
# the "normal" CC agent, the remainder would be the risk averse agent
agent_power_distribution_vec = [
        [0.3, 0.7],
        [0.5, 0.5],
        [0.7, 0.3],
    ]
cc_split_vec = [0.7, 0.8, 0.9]

fil_supply_discount_rate_vec = [5, 10, 15, 20, 25, 30]
filplus_agent_optimism_vec = [4]
normal_cc_agent_optimism_vec = [4]
risk_averse_cc_agent_optimism_vec = [4]

base_agent_discount_rate_yr_pct_vec = [25, 50]
normal_cc_agent_discount_rate_multiplier_vec = [1, 2]
risk_averse_cc_agent_discount_rate_multiplier_vec = [2, 3, 4, 5]
sdm_enable_date = date(2023, 10, 15) # ~6 months after the start of the simulation
sdm_slope_vec = [1.0, 0.285]

for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
    for renewal_rate in renewal_rate_vec:
        for agent_power_split in agent_power_distribution_vec:
            filp_agent_power = agent_power_split[0]
            remainder_power = 1 - filp_agent_power
            normal_cc_agent_power = remainder_power * agent_power_split[1]
            risk_averse_agent_power = remainder_power - normal_cc_agent_power
            
            # ensure that it sums to 1
            agent_power_distribution_control = np.asarray([filp_agent_power, remainder_power])
            agent_power_distribution_control = agent_power_distribution_control / np.sum(agent_power_distribution_control)
            agent_power_distribution_experiment = np.asarray([filp_agent_power, normal_cc_agent_power, risk_averse_agent_power])
            agent_power_distribution_experiment = agent_power_distribution_experiment / np.sum(agent_power_distribution_experiment)
            
            for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                for filplus_agent_optimism in filplus_agent_optimism_vec:
                    for base_agent_discount_rate_yr_pct in base_agent_discount_rate_yr_pct_vec:
                        filplus_agent_discount_rate = base_agent_discount_rate_yr_pct
                        for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                            for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * base_agent_discount_rate_yr_pct
                                for sdm_slope in sdm_slope_vec:
                                    filecoin_model_kwargs = exp_sdm.filecoin_model_kwargs(sdm_enable_date, sdm_slope)

                                    # baseline experiments
                                    name = 'SDMBaseline=%0.03f,FILP=%d,%d,%0.02f,CC=%d,%d,Onboard=%0.02f,RR=%0.02f,DR=%d' % \
                                        (
                                            sdm_slope,
                                            filplus_agent_optimism, filplus_agent_discount_rate, agent_power_distribution_control[0],
                                            normal_cc_agent_optimism, normal_cc_agent_discount_rate, 
                                            total_daily_rb_onboard_pib, renewal_rate, fil_supply_discount_rate,
                                         )
                                    name2experiment[name] = {
                                        'module_name': 'agentfil.cfg.exp_sdm',
                                        'instantiator': 'SDMBaselineExperiment',
                                        'instantiator_kwargs': {
                                            'max_sealing_throughput': C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                            'total_daily_onboard_rb_pib': total_daily_rb_onboard_pib,
                                            'renewal_rate': renewal_rate,
                                            'agent_power_distribution': agent_power_distribution_control,
                                            'fil_supply_discount_rate': fil_supply_discount_rate,
                                            'filplus_agent_optimism': filplus_agent_optimism,
                                            'filplus_agent_discount_rate_yr_pct': filplus_agent_discount_rate,
                                            'cc_agent_optimism': normal_cc_agent_optimism,
                                            'cc_agent_discount_rate_yr_pct': normal_cc_agent_discount_rate,

                                        },
                                        'filecoin_model_kwargs': filecoin_model_kwargs,
                                    }
                                    
                                    # test experiments
                                    for cc_split in cc_split_vec:
                                        for risk_averse_cc_agent_optimism in risk_averse_cc_agent_optimism_vec:
                                            for risk_averse_cc_agent_discount_rate_multiplier in risk_averse_cc_agent_discount_rate_multiplier_vec:
                                                risk_averse_cc_agent_discount_rate = risk_averse_cc_agent_discount_rate_multiplier * filplus_agent_discount_rate
                                                name = 'SDMExperiment=%0.03f,FILP=%d,%d,%0.02f,NormalCC=%d,%d,RACC=%d,%d,CCSplit=%0.02f,Onboard=%0.02f,RR=%0.02f,DR=%d' % \
                                                (
                                                    sdm_slope,
                                                    filplus_agent_optimism, filplus_agent_discount_rate, agent_power_distribution_experiment[0],
                                                    normal_cc_agent_optimism, normal_cc_agent_discount_rate, 
                                                    risk_averse_cc_agent_optimism, risk_averse_cc_agent_discount_rate,
                                                    cc_split, 
                                                    total_daily_rb_onboard_pib, 
                                                    renewal_rate, 
                                                    fil_supply_discount_rate,
                                                )

                                                name2experiment[name] = {
                                                    'module_name': 'agentfil.cfg.exp_sdm',
                                                    'instantiator': 'SDMExperiment',
                                                    'instantiator_kwargs': {
                                                        'max_sealing_throughput': C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                                        'total_daily_onboard_rb_pib': total_daily_rb_onboard_pib,
                                                        'renewal_rate': renewal_rate,
                                                        'agent_power_distribution': agent_power_distribution_experiment,
                                                        'fil_supply_discount_rate': fil_supply_discount_rate,
                                                        'filplus_agent_optimism': filplus_agent_optimism,
                                                        'filplus_agent_discount_rate_yr_pct': filplus_agent_discount_rate,
                                                        'normal_cc_agent_optimism': normal_cc_agent_optimism,
                                                        'normal_cc_agent_discount_rate_yr_pct': normal_cc_agent_discount_rate,
                                                        'riskaverse_cc_agent_optimism': risk_averse_cc_agent_optimism,
                                                        'riskaverse_cc_agent_discount_rate_yr_pct': risk_averse_cc_agent_discount_rate,

                                                    },
                                                    'filecoin_model_kwargs': filecoin_model_kwargs,
                                                }