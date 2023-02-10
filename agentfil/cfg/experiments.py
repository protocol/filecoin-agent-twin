"""
Add experiments to this file so that they can be run from the command line.
"""

name2experiment = {}

name2experiment['nagents-5_type-greedy_optimism-3'] = {
    'module_name': 'agentfil.cfg.exp_greedy_agents',
    'instantiator': 'ExpGreedyAgentsConstantOptimismTemplate',
    'instantiator_kwargs': {
        'num_agents': 5,
        'agent_optimism': 3
    }
}

name2experiment['nagents-5_type-greedy_optimism-4'] = {
    'module_name': 'agentfil.cfg.exp_greedy_agents',
    'instantiator': 'ExpGreedyAgentsConstantOptimismTemplate',
    'instantiator_kwargs': {
        'num_agents': 5,
        'agent_optimism': 4
    }
}

name2experiment['nagents-25_type-greedy_optimism-3'] = {
    'module_name': 'agentfil.cfg.exp_greedy_agents',
    'instantiator': 'ExpGreedyAgentsConstantOptimismTemplate',
    'instantiator_kwargs': {
        'num_agents': 25,
        'agent_optimism': 3
    }
}

name2experiment['nagents-25_type-greedy_optimism-4'] = {
    'module_name': 'agentfil.cfg.exp_greedy_agents',
    'instantiator': 'ExpGreedyAgentsConstantOptimismTemplate',
    'instantiator_kwargs': {
        'num_agents': 25,
        'agent_optimism': 4
    }
}

name2experiment['nagents-125_type-greedy_optimism-3'] = {
    'module_name': 'agentfil.cfg.exp_greedy_agents',
    'instantiator': 'ExpGreedyAgentsConstantOptimismTemplate',
    'instantiator_kwargs': {
        'num_agents': 125,
        'agent_optimism': 3
    }
}

name2experiment['nagents-125_type-greedy_optimism-4'] = {
    'module_name': 'agentfil.cfg.exp_greedy_agents',
    'instantiator': 'ExpGreedyAgentsConstantOptimismTemplate',
    'instantiator_kwargs': {
        'num_agents': 125,
        'agent_optimism': 4
    }
}