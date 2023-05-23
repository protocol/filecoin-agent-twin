#!/usr/bin/env python3

import argparse
import os
import importlib
from datetime import datetime, timedelta
from tqdm.auto import tqdm

from agentfil import constants
from agentfil.cfg import experiments
from agentfil.filecoin_model import FilecoinModel

def run_experiment(auth_config, experiment_name, experiment_output_dir, 
                   start_date, end_date, 
                   verbose=False):
    os.makedirs(experiment_output_dir, exist_ok=True)

    experiment = experiments.name2experiment[experiment_name]
    module_name = experiment['module_name']
    class_name = experiment['instantiator']
    kwargs = experiment['instantiator_kwargs']
    filecoin_model_kwargs = experiment.get('filecoin_model_kwargs', {})
    module = importlib.import_module(module_name)
    experiment_obj = getattr(module, class_name)(**kwargs)

    agent_types, agent_kwargs_vec, agent_power_distributions = experiment_obj.get_agent_cfg()
    
    num_agents = len(agent_types)
    minting_process_kwargs = experiment_obj.get_minting_process_cfg()
    price_process_kwargs = experiment_obj.get_price_process_cfg()
    fil_supply_discount_rate_process_kwargs = experiment_obj.get_fil_supply_discount_rate_process_cfg()
    
    forecast_length = (end_date - start_date).days

    filecoin_model = FilecoinModel(num_agents, start_date, end_date, 
                                   spacescope_cfg=auth_config,
                                   agent_types=agent_types, agent_kwargs_list=agent_kwargs_vec, 
                                   agent_power_distributions=agent_power_distributions,
                                   compute_cs_from_networkdatastart=True, use_historical_gas=False,
                                   price_process_kwargs=price_process_kwargs,
                                   rewards_per_sector_process_kwargs=minting_process_kwargs,
                                   fil_supply_discount_rate_process_kwargs=fil_supply_discount_rate_process_kwargs,
                                   **filecoin_model_kwargs)
    for _ in tqdm(range(forecast_length), disable=not verbose):
        filecoin_model.step()

    # save data to the output directory
    filecoin_model.save_data(experiment_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auth-config', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, default='default')
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--start-date', type=str, default='today')
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    if args.experiment_name not in experiments.name2experiment:
        raise ValueError('Experiment name not found: {}'.format(args.experiment_name))
    
    if args.start_date == 'today':
        start_date = datetime.today().date() - timedelta(days=2)
    else:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    if args.end_date is None:
        forecast_length = 365*5
        end_date = start_date + timedelta(days=forecast_length)
    else:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    run_experiment(args.auth_config, args.experiment_name, args.output_dir, 
                   start_date, end_date, args.verbose)