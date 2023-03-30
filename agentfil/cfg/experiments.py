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
mechanism.  See the 
"""
num_agents_vec = [1]
max_daily_rb_onboard_pib_vec = [5, 10, 25]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3, 360*5]
min_discount_rate_pct_vec = [10, 15, 20]
max_discount_rate_pct_vec = [50, 75, 100]
start_discount_rate_pct = 0 # a noop, will be overwritten

for num_agents in num_agents_vec:
    for max_daily_rb_onboard_pib in max_daily_rb_onboard_pib_vec:
        for renewal_rate in renewal_rate_vec:
            for fil_plus_rate in fil_plus_rate_vec:
                for sector_duration in sector_duration_vec:
                    for min_discount_rate_pct in min_discount_rate_pct_vec:
                        for max_discount_rate_pct in max_discount_rate_pct_vec:
                            name = 'DCA-LinearAdaptiveFilSupplyDiscountRate=[%d,%d]-MaxDailyOnboard=%0.02f-RenewalRate=%0.02f-FilPlusRate=%0.02f-SectorDuration=%d' % \
                                (min_discount_rate_pct, max_discount_rate_pct, max_daily_rb_onboard_pib, renewal_rate, fil_plus_rate, sector_duration)
                            name2experiment[name] = {
                                'module_name': 'agentfil.cfg.exp_dca_agents',
                                'instantiator': 'ExpDCAAgentsLinearAdaptiveDiscountRate',
                                'instantiator_kwargs': {
                                    'num_agents': num_agents,
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
Experiments that begin to add complexity to the baseline. Lets term these "Hybrid" experiments.  Here
a defined % of the network power is assigned to DCA agents, which provide a "steady-state" behavior. 
The remaining % of the network power is assigned to, in this specific case, the Basic Rational agents.
We seek to understand how the proportion of steady-state to dynamic agents affects:
  1) Rewards for both types of agents
  2) The overall network econometrics
"""
num_agents_vec = [2]
max_daily_rb_onboard_pib_vec = [5, 10, 25]
renewal_rate_vec = [.4, .6, .8]
fil_plus_rate_vec = [.4, .6, .8]
sector_duration_vec = [360, 360*3, 360*5]
steady_state_total_network_power_vec = [.85, .90, .95]

