"""
Add experiments to this file so that they can be run from the command line.
"""

name2experiment = {}

num_agents_vec = [5, 50]
agent_optimism_vec = [2, 3, 4]
max_agent_power_vec = [0.2, 0.3, 0.4]
minmax_optimism_cfgs_vec = [(2,4), (1,5), (3,5)]

### Define some Greedy Agent Experiments
for num_agents in num_agents_vec:
    for agent_optimism in agent_optimism_vec:
        name = f'nagents-%d_type-greedy_optimism-%d_uniformpowerdist' % (num_agents, agent_optimism)
        name2experiment[name] = {
            'module_name': 'agentfil.cfg.exp_greedy_agents',
            'instantiator': 'ExpGreedyAgentsConstantOptimismUniformPowerDistribution',
            'instantiator_kwargs': {
                'num_agents': num_agents,
                'agent_optimism': agent_optimism
            }
        }

for max_agent_power in max_agent_power_vec:
    for num_agents in num_agents_vec:
        for agent_optimism in agent_optimism_vec:
            name = f'nagents-%d_type-greedy_optimism-%d_maxpow-%0.02f_geometricdist' % (num_agents, agent_optimism, max_agent_power)
            name2experiment[name] = {
                'module_name': 'agentfil.cfg.exp_greedy_agents',
                'instantiator': 'ExpGreedyAgentsConstantOptimismGeometricPowerDistribution',
                'instantiator_kwargs': {
                    'num_agents': num_agents,
                    'agent_optimism': agent_optimism,
                    'max_agent_power': max_agent_power
                }
            }

for minmax_optimism_cfg in minmax_optimism_cfgs_vec:
    min_optimism, max_optimism = minmax_optimism_cfg
    for num_agents in num_agents_vec:
        for max_agent_power in max_agent_power_vec:
            name = f'nagents-%d_type-greedy_optimism-proportional-min-%d-max-%d_maxpow-%0.02f_geometricdist' % (num_agents, min_optimism, max_optimism, max_agent_power)
            name2experiment[name] = {
                'module_name': 'agentfil.cfg.exp_greedy_agents',
                'instantiator': 'ExpGreedyAgentsProportionalOptimismGeometricPowerDistribution',
                'instantiator_kwargs': {
                    'num_agents': num_agents,
                    'max_agent_power': max_agent_power,
                    'min_optimism': min_optimism,
                    'max_optimism': max_optimism
                }
            }

### Define some Far-Sighted Agent Experiments
for num_agents in num_agents_vec:
    for agent_optimism in agent_optimism_vec:
        name = f'nagents-%d_type-farsighted_optimism-%d_uniformpowerdist' % (num_agents, agent_optimism)
        name2experiment[name] = {
            'module_name': 'agentfil.cfg.exp_farsighted_agents',
            'instantiator': 'ExpFarsightedAgentsConstantOptimismUniformPowerDistribution',
            'instantiator_kwargs': {
                'num_agents': num_agents,
                'agent_optimism': agent_optimism,
                'far_sightedness_days': 90,
                'reestimate_every_days': 90
            }
        }

for max_agent_power in max_agent_power_vec:
    for num_agents in num_agents_vec:
        for agent_optimism in agent_optimism_vec:
            name = f'nagents-%d_type-farsighted_optimism-%d_maxpow-%0.02f_geometricdist' % (num_agents, agent_optimism, max_agent_power)
            name2experiment[name] = {
                'module_name': 'agentfil.cfg.exp_farsighted_agents',
                'instantiator': 'ExpFarsightedAgentsConstantOptimismGeometricPowerDistribution',
                'instantiator_kwargs': {
                    'num_agents': num_agents,
                    'agent_optimism': agent_optimism,
                    'far_sightedness_days': 90,
                    'reestimate_every_days': 90,
                    'max_agent_power': max_agent_power
                }
            }

for minmax_optimism_cfg in minmax_optimism_cfgs_vec:
    min_optimism, max_optimism = minmax_optimism_cfg
    for num_agents in num_agents_vec:
        for max_agent_power in max_agent_power_vec:
            name = f'nagents-%d_type-farsighted_optimism-proportional-min-%d-max-%d_maxpow-%0.02f_geometricdist' % (num_agents, min_optimism, max_optimism, max_agent_power)
            name2experiment[name] = {
                'module_name': 'agentfil.cfg.exp_farsighted_agents',
                'instantiator': 'ExpFarsightedAgentsProportionalOptimismGeometricPowerDistribution',
                'instantiator_kwargs': {
                    'num_agents': num_agents,
                    'max_agent_power': max_agent_power,
                    'min_optimism': min_optimism,
                    'max_optimism': max_optimism,
                    'far_sightedness_days': 90,
                    'reestimate_every_days': 90,
                }
            }
