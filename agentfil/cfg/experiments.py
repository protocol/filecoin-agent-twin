import numpy as np
import constants as C

from ..agents import dca_agent, basic_rational_agent

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
steady_state_total_network_power_vec = [.85, .90, .95]
fil_supply_discount_rate_vec = [20, 25, 30]

for steady_state_total_network_power in steady_state_total_network_power_vec:
    for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for fil_plus_rate in fil_plus_rate_vec:
                for sector_duration in sector_duration_vec:
                    for fil_supply_discount_rate in fil_supply_discount_rate_vec:
                        agent_types = [
                            dca_agent.DCAAgent,
                            basic_rational_agent.BasicRationalAgent
                        ]
                        # 1 - since we are adding rational agents on top of steady-state, we don't scale
                        #     their max sealing throughput. The assumption is that the agent is already being
                        #     rational, so their computing capabilities are high and they have a high sealing throughput.
                        #     This can be easily changed if we want to explore the impact of rational agents with
                        #     less sealing throughput, can their relative gains be higher since they act rationally??
                        # 2 - we also don't scale the agents daily onboarding. It's easy to justify both sides of the 
                        #     coin here, so it might be useful to try it with and without some scaling.
                        agent_kwargs = [
                            {
                                'max_sealing_throughput':C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                'max_daily_rb_onboard_pib': max_daily_rb_onboard_pib,
                                'renewal_rate': renewal_rate,
                                'fil_plus_rate': fil_plus_rate,
                                'sector_duration': sector_duration,
                            },
                            {
                                'max_sealing_throughput':C.DEFAULT_MAX_SEALING_THROUGHPUT_PIB,
                                'max_daily_rb_onboard_pib': max_daily_rb_onboard_pib,
                                'renewal_rate': renewal_rate,
                                'fil_plus_rate': fil_plus_rate,
                                'sector_duration': sector_duration,
                                'discount_rate_floor_pct': min(fil_supply_discount_rate_vec)
                            }
                        ]
                        dca_agent_power_frac = steady_state_total_network_power
                        basic_rational_agent_power_frac = 1 - steady_state_total_network_power
                        agent_power_distribution = [dca_agent_power_frac, basic_rational_agent_power_frac]
                        
                        name = 'DCA=%0.02f-BasicRational=%0.02f-ConstFilSupplyDiscountRate=%d-MaxDailyOnboard=%0.02f,%0.02f-RenewalRate=%0.02f,%0.02f-FilPlusRate=%0.02f,%0.02f-SectorDuration=%d,%d' % \
                            (dca_agent_power_frac, 
                             basic_rational_agent_power_frac, 
                             fil_supply_discount_rate, 
                             max_daily_rb_onboard_pib, 
                             max_daily_rb_onboard_pib, 
                             renewal_rate, 
                             renewal_rate, 
                             fil_plus_rate, 
                             fil_plus_rate, 
                             sector_duration,
                             sector_duration
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

