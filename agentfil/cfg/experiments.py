import numpy as np

from datetime import date, timedelta

from .. import constants as C
from ..agents import dca_agent, basic_rational_agent, roi_agent, npv_agent
from . import exp_sdm
from . import exp_miner_proportion
from . import exp_locktarget

"""
Add experiments to this file so that they can be run from the command line.
"""

name2experiment = {}

"""
Baseline experiments - establish the baseline by onboarding:
    - a constant amount of power
    - a constant renewal rate
    - a constant FIL+ rate
    - 0 % external discount rate (i.e. lending is free)
    - a constant sector duration of 360 days
"""
max_daily_rb_onboard_pib_vec = [4, 6, 8]
renewal_rate_vec = [0.4, 0.5, 0.6, 0.7, 0.8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration = 360
fil_supply_discount_rate = 0
max_sealing_throughput = [C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB]
for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
    for renewal_rate in renewal_rate_vec:
        for fil_plus_rate in fil_plus_rate_vec:
            name = 'BaselineDCA_RBP_%0.02f-RR_%0.02f-FPR_%0.02f-Dur_%0.02f' % \
                (max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
            name2experiment[name] = {
                'module_name': 'agentfil.cfg.exp_dca_agents',
                'instantiator': 'ExpDCAAgentsConstantDiscountRate',
                'instantiator_kwargs': {
                    'num_agents': 1,
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

# max_daily_rb_onboard_pib_vec = [4,6,8,50]
max_total_onboard_pib_vec = [6,50]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3, 360*5]
fil_supply_discount_rate_vec = [10, 20, 30, 40, 50]

for num_agents in num_agents_vec:
    for agent_power_distribution in agent_power_distribution_vec:
        # normalize to sum to one
        agent_power_distribution_normalized = agent_power_distribution/np.sum(agent_power_distribution)
        # setup the max sealing throughput to be proportional to agent power, capped by the default max
        # max_sealing_throughput_scaled = np.sum(agent_power_distribution)/max(agent_power_distribution) * C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB
        # max_sealing_throughput = agent_power_distribution_normalized * max_sealing_throughput_scaled


        for max_total_onboard_pib in max_total_onboard_pib_vec:
            for renewal_rate in renewal_rate_vec:
                for fil_plus_rate in fil_plus_rate_vec:
                    for sector_duration in sector_duration_vec:
                        for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                            agent_power_distribution_str = ','.join([str(x) for x in agent_power_distribution])
                            name = 'DCAPowerConcentration=%s-ConstFilSupplyDiscountRate=%d-MaxDailyOnboard=%0.02f-RenewalRate=%0.02f-FilPlusRate=%0.02f-SectorDuration=%d' % \
                                (agent_power_distribution_str,fil_supply_discount_rate, max_total_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                            name2experiment[name] = {
                                'module_name': 'agentfil.cfg.exp_dca_agents',
                                'instantiator': 'ExpDCAAgentsPowerScaledConstantDiscountRate',
                                'instantiator_kwargs': {
                                    'num_agents': num_agents,
                                    'agent_power_distribution': agent_power_distribution_normalized,
                                    'agent_max_sealing_throughput': C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                    'max_daily_rb_onboard_pib': max_total_onboard_pib,
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
filplus_agent_optimism_vec = [3,4,5]
normal_cc_agent_optimism_vec = [3,4,5]
risk_averse_cc_agent_optimism_vec = [3,4,5]

filplus_agent_discount_rate_yr_pct_vec = [25, 50, 75]
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
                    for base_agent_discount_rate_yr_pct in filplus_agent_discount_rate_yr_pct_vec:
                        filplus_agent_discount_rate = base_agent_discount_rate_yr_pct
                        for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                            for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * base_agent_discount_rate_yr_pct
                                for sdm_slope in sdm_slope_vec:
                                    filecoin_model_kwargs = exp_sdm.filecoin_model_kwargs(sdm_enable_date, sdm_slope)

                                    # baseline experiments
                                    name = 'SDMBaseline_%0.03f,FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
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
                                                risk_averse_cc_agent_discount_rate = risk_averse_cc_agent_discount_rate_multiplier * base_agent_discount_rate_yr_pct
                                                name = 'SDMExperiment_%0.03f,FILP_%d,%d,%0.02f,NormalCC_%d,%d,RACC_%d,%d,CCSplit_%0.02f,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
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

"""
Experiments that relate to the assessing sensitivity of the network to 
the proportion of FIL+ agents in the network - conducted at the subpopulation level.
"""
total_daily_rb_onboard_pib_vec = [4, 6, 8]
renewal_rate_vec = [0.5, 0.6, 0.7]
filp_agent_power_distribution_vec = np.arange(0.3, 0.7+0.1, 0.1)

fil_supply_discount_rate_vec = [10, 20, 30]
filplus_agent_optimism_vec = [4]
normal_cc_agent_optimism_vec = [4]

filplus_agent_discount_rate_yr_pct_vec = [25, 50]
normal_cc_agent_discount_rate_multiplier_vec = [1, 2]

for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
    for renewal_rate in renewal_rate_vec:
        for filp_agent_power in filp_agent_power_distribution_vec:
            for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                for filplus_agent_optimism in filplus_agent_optimism_vec:
                    for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                        for filplus_agent_discount_rate in filplus_agent_discount_rate_yr_pct_vec:
                            for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * filplus_agent_discount_rate
                                agent_power_distribution = np.asarray([filp_agent_power, 1 - filp_agent_power])
                                agent_power_distribution = agent_power_distribution / np.sum(agent_power_distribution)

                                name = 'MinerProportionSensitivity,FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                    (
                                        filplus_agent_optimism, filplus_agent_discount_rate, filp_agent_power,
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
                                            'agent_power_distribution': agent_power_distribution,
                                            'fil_supply_discount_rate': fil_supply_discount_rate,
                                            'filplus_agent_optimism': filplus_agent_optimism,
                                            'filplus_agent_discount_rate_yr_pct': filplus_agent_discount_rate,
                                            'cc_agent_optimism': normal_cc_agent_optimism,
                                            'cc_agent_discount_rate_yr_pct': normal_cc_agent_discount_rate,

                                        },
                                        'filecoin_model_kwargs': {},  # do not add any policy changes, default model
                                    }
                                
"""
Lock Target experiments
TODO: more description!
"""
total_daily_rb_onboard_pib_vec = [4, 6, 8]
renewal_rate_vec = [0.5, 0.6, 0.7]
filp_agent_power_distribution_vec = np.arange(0.3, 0.7+0.1, 0.1)

fil_supply_discount_rate_vec = [10, 20, 30]
filplus_agent_optimism_vec = [4]
normal_cc_agent_optimism_vec = [4]

filplus_agent_discount_rate_yr_pct_vec = [25, 50]
normal_cc_agent_discount_rate_multiplier_vec = [1, 2]

lock_target_increase_value_vec = [0.4, 0.5, 0.6, 0.7, 0.8]
lock_target_increase_dynamics_vec = ['linear_ramp', 'jump']
lock_target_ramp_speed_months_vec = [3, 6, 12]
lock_target_increase_date_start = date(2023, 10, 15)

for total_daily_rb_onboard_pib in total_daily_rb_onboard_pib_vec:
    for renewal_rate in renewal_rate_vec:
        for filp_agent_power in filp_agent_power_distribution_vec:
            for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                for filplus_agent_optimism in filplus_agent_optimism_vec:
                    for normal_cc_agent_optimism in normal_cc_agent_optimism_vec:
                        for filplus_agent_discount_rate in filplus_agent_discount_rate_yr_pct_vec:
                            for normal_cc_agent_discount_rate_multiplier in normal_cc_agent_discount_rate_multiplier_vec:
                                normal_cc_agent_discount_rate = normal_cc_agent_discount_rate_multiplier * filplus_agent_discount_rate

                                for lock_target_increase_value in lock_target_increase_value_vec:
                                    for lock_target_increase_dynamics in lock_target_increase_dynamics_vec:
                                        if lock_target_increase_dynamics == 'jump':
                                            # configure the lock target change policy
                                            lock_target_callable, lock_target_callable_kwargs = \
                                                exp_locktarget.get_lock_target_post_step_callable(
                                                    lock_target_increase_dynamics, 
                                                    lock_target_increase_value,
                                                    lock_target_increase_date_start,
                                                    None
                                                )
                                            filecoin_model_kwargs = {
                                                'user_post_network_update_callables': [lock_target_callable],
                                                'user_post_network_update_callables_kwargs_list': [lock_target_callable_kwargs],
                                            }
                                            
                                            name = 'LockTarget[%0.1f,%s],FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                                (
                                                    lock_target_increase_value, lock_target_increase_dynamics,
                                                    filplus_agent_optimism, filplus_agent_discount_rate, filp_agent_power,
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
                                            
                                        elif lock_target_increase_dynamics == 'linear_ramp':
                                            for lock_target_ramp_speed_months in lock_target_ramp_speed_months_vec:
                                                lock_target_increase_date_stop = lock_target_increase_date_start + timedelta(days=30*lock_target_ramp_speed_months)
                                                lock_target_callable, lock_target_callable_kwargs = \
                                                    exp_locktarget.get_lock_target_post_step_callable(
                                                        lock_target_increase_dynamics, 
                                                        lock_target_increase_value,
                                                        lock_target_increase_date_start,
                                                        lock_target_increase_date_stop
                                                    )
                                                filecoin_model_kwargs = {
                                                    'user_post_network_update_callables': [lock_target_callable],
                                                    'user_post_network_update_callables_kwargs_list': [lock_target_callable_kwargs],
                                                }

                                                name = 'LockTarget[%0.1f,%s,%d],FILP_%d,%d,%0.02f,CC_%d,%d,Onboard_%0.02f,RR_%0.02f,DR_%d' % \
                                                    (
                                                        lock_target_increase_value, lock_target_increase_dynamics, lock_target_ramp_speed_months,
                                                        filplus_agent_optimism, filplus_agent_discount_rate, filp_agent_power,
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
                                                

"""
Switching Agent experiments
"""
fil_supply_discount_rate_vec = [10, 20, 30]

# common to all agents
max_daily_rb_onboard_pib_vec = [6]
renewal_rate_vec = [.6]
fil_plus_rate_vec = [.6]

# switching configurations
random_seed = 1234
switching_period_vec = [30, 90, 180]
switching_strategies_vec = [['dca', 'npv', 'roi']]
switching_strategy_probabilities_vec = [[0.33, 0.33, 0.34]]

# keep all of these static, since we're interested in parametrizing the switching strategies
# dca agent specific config
sector_duration_vec = [360]

# npv agent configs
npv_agent_optimism_vec = [4]
npv_agent_discount_rate_yr_pct_vec = [50]

# ROI agent configs
roi_agent_optimism_vec = [4]
roi_threshold_vec = [0.1]

for fil_supply_discount_rate in fil_supply_discount_rate_vec:
    for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for fil_plus_rate in fil_plus_rate_vec:
                for switching_period in switching_period_vec:
                    for switching_strategy in switching_strategies_vec:
                        for switching_strategy_probabilities in switching_strategy_probabilities_vec:
                            for sector_duration in sector_duration_vec:
                                for npv_agent_optimism in npv_agent_optimism_vec:
                                    for npv_agent_discount_rate_yr_pct in npv_agent_discount_rate_yr_pct_vec:
                                        for roi_agent_optimism in roi_agent_optimism_vec:
                                            for roi_threshold in roi_threshold_vec:
                                                name = 'Switching-0.33DCA-0.33NPV-0.34ROI-MaxRBP_%0.02f-RR_%0.02f-FPR_%0.02f-SR_%d-Dur_%d-NPV_%d_%d-ROI_%d_%0.02f-DR_%d' % \
                                                    (max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, switching_period, sector_duration, 
                                                     npv_agent_optimism, npv_agent_discount_rate_yr_pct, roi_agent_optimism, roi_threshold, fil_supply_discount_rate)
                                                name2experiment[name] = {
                                                    'module_name': 'agentfil.cfg.exp_switching_agents',
                                                    'instantiator': 'ExpSwitchingAgentsConstantDiscountRate',
                                                    'instantiator_kwargs': {
                                                        'num_agents':1, 
                                                        'agent_power_distribution':[1],
                                                        'agent_max_sealing_throughput':C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                                        'max_daily_rb_onboard_pib':max_daily_rb_onboard_pib, 
                                                        'renewal_rate':renewal_rate, 
                                                        'fil_plus_rate':fil_plus_rate, 
                                                        'fil_supply_discount_rate':fil_supply_discount_rate,
                                                        'random_seed':random_seed,
                                                        'switching_period_days':switching_period, 
                                                        'switching_strategies':switching_strategy, 
                                                        'switching_strategy_probabilities':switching_strategy_probabilities,
                                                        'dca_sector_duration':sector_duration,
                                                        'npv_agent_optimism':npv_agent_optimism, 
                                                        'npv_agent_discount_rate_yr_pct':npv_agent_discount_rate_yr_pct,
                                                        'roi_agent_optimism':roi_agent_optimism, 
                                                        'roi_threshold':roi_threshold
                                                    },
                                                    'filecoin_model_kwargs': {},  # do not add any policy changes, default model
                                                }

"""
Shock experiments
"""
population_power_breakdown = [
    [0.33, 0.33, 0.34],
    [0.495, 0.495, 0.01],
    [0.695, 0.295, 0.01],
]
subpopulation_terminate_pcts = [0.0, 0.3, 0.5, 0.7]
terminate_date = date(2023, 11, 1)

total_onboard_rbp = 6  # across all agents in the simulation
renewal_rate = 0.6     # for agents which decide to stay on the network
fil_plus_rate = 0.8    # for the mixed agents which decide to stay on the network
sector_duration = 360
num_agents = 3
fil_supply_discount_rate_vec = [10, 20, 30]

for population_power in population_power_breakdown:
    agent_power_distribution = population_power
    for subpopulation_terminate_pct in subpopulation_terminate_pcts:
        for fil_supply_discount_rate in fil_supply_discount_rate_vec:
            name = 'Terminate_%0.02f-FP_%0.02f-CC_%0.02f-MX_%0.02f-MaxRBP_%0.02f-RR_%0.02f-FPR_%0.02f-DR_%d' % \
                (subpopulation_terminate_pct, 
                 agent_power_distribution[0], agent_power_distribution[1], agent_power_distribution[2],
                 total_onboard_rbp, renewal_rate, fil_plus_rate, fil_supply_discount_rate)
            name2experiment[name] = {
                'module_name': 'agentfil.cfg.exp_dca_terminate',
                'instantiator': 'ExpDCAAgentsTerminate',
                'instantiator_kwargs': {
                    'num_agents':num_agents, 
                    'agent_power_distribution':agent_power_distribution,
                    'subpopulation_terminate_pct':subpopulation_terminate_pct,
                    'agent_max_sealing_throughput':C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,  # a noop
                    'max_daily_rb_onboard_pib':total_onboard_rbp, 
                    'renewal_rate':renewal_rate, 
                    'fil_plus_rate':fil_plus_rate, 
                    'sector_duration': sector_duration,
                    'fil_supply_discount_rate':fil_supply_discount_rate,
                    'terminate_date': terminate_date,
                },
                'filecoin_model_kwargs': {},  # do not add any policy changes, default model
            }

"""
Shock w/ ROI agents
"""
population_power_breakdown = [
    [0.33, 0.33, 0.34],
    [0.495, 0.495, 0.01],
    [0.695, 0.295, 0.01],
    [0.295, 0.695, 0.01],
]
subpopulation_terminate_pcts = [0.3, 0.5, 0.7]
terminate_date = date(2023, 11, 1)

total_min_onboard_rbp = 0
total_max_onboard_rbp_vec = [3,6,15]
min_rr = 0.0
max_rr_vec = [0.4, 0.8]
min_roi_vec = [0.1, 0.2, 0.3]
max_roi_vec = [0.8, 0.9, 1.0]
roi_agent_optimism_vec = [2,3,4]
fil_plus_rate = 0.8    # for the mixed agents which decide to stay on the network
sector_duration = 360
num_agents = 3
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
                            name2experiment[name] = {
                                'module_name': 'agentfil.cfg.exp_roi_terminate',
                                'instantiator': 'ExpROIAdaptDCATerminate',
                                'instantiator_kwargs': {
                                    'num_agents':num_agents, 
                                    'agent_power_distribution':agent_power_distribution,
                                    'subpopulation_terminate_pct':subpopulation_terminate_pct,
                                    'max_sealing_throughput':C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,

                                    'min_daily_rb_onboard_pib':total_min_onboard_rbp,
                                    'max_daily_rb_onboard_pib':total_max_onboard_rbp,
                                    'min_renewal_rate':min_rr,
                                    'max_renewal_rate':max_rr,
                                    'fil_plus_rate':fil_plus_rate,
                                    'min_roi':min_roi,
                                    'max_roi':max_roi,
                                    'roi_agent_optimism':roi_agent_optimism,

                                    'sector_duration': sector_duration,
                                    'fil_supply_discount_rate':fil_supply_discount_rate,
                                    'terminate_date': terminate_date,
                                },
                            }