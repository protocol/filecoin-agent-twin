from typing import Dict, Any, Tuple, List
from abc import ABC, abstractmethod

class ExperimentCfg(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_agent_cfg(self) -> Tuple[List, List, List]:
        """
        Code to configure the agents. Must return a tuple of 3 items:
        1. A list of agent constructors
        2. A list of agent constructor kwargs
        3. A list of the agent power fraction (proxy of agent size)
        """
        pass

    def get_rewards_per_sector_process_cfg(self) -> Dict:
        """
        Code to configure the minting process. Must return a dictionary of kwargs 
        will be passed to the minting process constructor.
        """
        minting_process_kwargs = {
            'forecast_history': 180,
            'update_every_days': 90,
            'num_warmup_mcmc': 500,
            'num_samples_mcmc': 500,
            'seasonality_mcmc': 1000,
            'num_chains_mcmc': 2,
            'verbose': False,
            'keep_previous_predictions': False,
            'keep_power_predictions': False,
        }
        return minting_process_kwargs

    def get_price_process_cfg(self):
        """
        Code to configure the price process. Must return a dictionary of kwargs
        will be passed to the price process constructor.
        """
        price_process_kwargs = {
            'forecast_num_mc':1000,
            'random_seed':1234
        }
        return price_process_kwargs

    def get_capital_inflow_process_cfg(self):
        """
        Code to configure the capital inflow process. Must return a dictionary of kwargs
        will be passed to the capital inflow process constructor.
        """
        capital_inflow_process_kwargs = {
            'debug_model': False
        }
        return capital_inflow_process_kwargs
    
    def get_fil_supply_discount_rate_process_cfg(self):
        """
        Code to configure the FIL supply discount rate process. Must return a dictionary of kwargs
        will be passed to the FIL supply discount rate process constructor.
        """
        fil_supply_discount_rate_process_kwargs = {
            'min_discount_rate_pct':0, 
            'max_discount_rate_pct':200,
            'start_discount_rate_pct':25,
            'behavior':'constant',
            'seed':1234
        }
        return fil_supply_discount_rate_process_kwargs