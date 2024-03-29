from datetime import timedelta
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power
from .filecoin_model import apply_qa_multiplier

import numpy as np
import pandas as pd

class GreedyAgent(SPAgent):
    """
    Greedy Agent Seeding
    --------------------
    The agent is seeded with a historical power profile from the start of network data (2021-03-15)
    to the simulation start date. Over that time, the amount of rewards the agent has received from
    the network, minus the amount of FIL it has pledged, burned due to terminations, etc is all accounted
    for in the agent's accounting. At the start of simulation, the agent has a certain amount of FIL
    in its wallet, and a certain amount of power on the network based on the historical power profile.

    It can use the remaining FIL to onboard more power according to its behavior, described below.
    TLDR: This agent approximates locally optimal behavior but better strategies exist.

    Agent Logic/Behavior
    --------------------
    duration_vec_days = [12, 36]*30
    profitability_vec = []
    for d in duration_vec_days:
        - Estimate ROI
            - x = Estimate future rewards / sector # currently linear extrapolation of last day of data for duration d
            - roi = (np.sum(x)/pledge_per_sector)
            - duration_yrs = d/360
            - roi_annualized = (1+roi)^(1/duration_yrs) - 1
        - Estimate future USD/FIL exchange rate at time t+d
        - profit_metric = (1+roi_annualized)*exchange_rate[t+d] - exchange_rate[t]
        - profitability_vec.append(profit_metric)
    - max_profit_idx = argmax(profitability_vec)
    - best duration = duration_vec_days[max_profit_idx]
    if profitability_vec[max_profit_idx] > 0: power for duration = best duration
        - [X] Add max possible deal power (based on available FIL)
        - [ ] With remaining FIL, renew what is scheduled to expire
        - [ ] If still further remaining FIL, onboard CC power

    TODO:
    [X] - Iterative estimation of block rewards / day
    [ ] - How to model external macro-environmental factors (interest rates, etc.) in order to
          determine if agent should borrow FIL to onboard more power than it has reserves for?

    """

    def __init__(self, model, id, historical_power, start_date, end_date, 
                 random_seed=1111, agent_optimism=3):
        """
        Args:
            model: the model object
            id: the agent id
            historical_power: the historical power of the agent
            start_date: the start date of the simulation
            end_date: the end date of the simulation
            accounting_df: a dataframe with two columns, date and USD. 
                           Currently, the USD field is not used, but will be modified in the future to model interest rates, etc.    
            random_seed: the random seed for the agent
            fil_usd_price_optimism_scale: integer between 1 and 5 representing the optimism of the agent, 
                                          1 being most pessimistic and 5 being most optimistic
        """
        super().__init__(model, id, historical_power, start_date, end_date)
        
        self.random_seed = random_seed
        self.duration_vec_days = (np.asarray([12, 36])*30).astype(np.int32)
        self.agent_optimism = agent_optimism
        self.validate()

        self.map_optimism_scales()

        # additional items to track for the greedy agent
        for d in self.duration_vec_days:
            self.agent_info_df[f'roi_estimate_{d}'] = 0
            self.agent_info_df[f'profit_metric_{d}'] = 0
        

    def map_optimism_scales(self):
        self.optimism_to_price_quantile_str = {
            1 : "Q05",
            2 : "Q25",
            3 : "Q50",
            4 : "Q75",
            5 : "Q95"
        }
        self.optimism_to_dayrewardspersector_quantile_str = {
            1 : "Q05",
            2 : "Q25",
            3 : "Q50",
            4 : "Q75",
            5 : "Q95"
        }
        

    def validate(self):
        assert self.agent_optimism >= 1 and self.agent_optimism <= 5, \
                "optimism must be an integer between 1 and 5"
        assert type(self.agent_optimism) == int, "agent_optimism must be an integer"

    def forecast_day_rewards_per_sector(self, forecast_start_date, forecast_length):
        k = 'day_rewards_per_sector_forecast_' + self.optimism_to_dayrewardspersector_quantile_str[self.agent_optimism]
        start_idx = self.model.global_forecast_df[pd.to_datetime(self.model.global_forecast_df['date']) == pd.to_datetime(forecast_start_date)].index[0]
        end_idx = start_idx + forecast_length
        future_rewards_per_sector = self.model.global_forecast_df.loc[start_idx:end_idx, k].values
        
        return future_rewards_per_sector


    def estimate_roi(self, sector_duration, date_in):
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]

        # NOTE: we need to use yesterday's metrics b/c today's haven't yet been aggregated by the system yet
        prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']

        # TODO: make this an iterative update rather than full-estimate every day
        future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        roi_estimate = future_rewards_per_sector_estimate.sum() / prev_day_pledge_per_QAP
        
        # annualize it so that we can have the same frame of reference when comparing different sector durations
        duration_yr = sector_duration / 360.0  
        roi_estimate_annualized = (1.0+roi_estimate)**(1.0/duration_yr) - 1
        
        # if np.isnan(future_rewards_per_sector_estimate.sum()) or np.isnan(prev_day_pledge_per_QAP) or np.isnan(roi_estimate) or np.isnan(roi_estimate_annualized):
        #     print(self.unique_id, future_rewards_per_sector_estimate.sum(), prev_day_pledge_per_QAP, roi_estimate, roi_estimate_annualized)

        return roi_estimate_annualized

    def get_exchange_rate(self, date_in):
        key = 'price_' + self.optimism_to_price_quantile_str[self.agent_optimism]
        v = self.model.global_forecast_df.loc[self.model.global_forecast_df['date'] == date_in, key].values
        if len(v) > 0:
            return v[0]
        else:
            # return last known prediction.  Agent can do better than this if they choose to
            return self.model.global_forecast_df.iloc[-1][key]


    def step(self):
        roi_estimate_vec = []
        profitability_vec = []

        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(self.current_date)].index[0]
        current_exchange_rate = self.get_exchange_rate(self.current_date)
        for d in self.duration_vec_days:    
            roi_estimate = self.estimate_roi(d, self.current_date)
            
            # Estimate future USD/FIL exchange rate at time t+d
            future_date = self.current_date + timedelta(days=int(d))
            future_exchange_rate = self.get_exchange_rate(future_date)
        
            profit_metric = (1+roi_estimate)*future_exchange_rate - current_exchange_rate
            
            roi_estimate_vec.append(roi_estimate)
            profitability_vec.append(profit_metric)

            self.agent_info_df.loc[agent_df_idx, 'roi_estimate_%d' % (d,)] = roi_estimate
            self.agent_info_df.loc[agent_df_idx, 'profit_metric_%d' % (d,)] = profit_metric
        
        max_profit_idx = np.argmax(profitability_vec)
        best_duration = self.duration_vec_days[max_profit_idx]
        if profitability_vec[max_profit_idx] > 0: 
            max_possible_qa_power = self.get_max_onboarding_qap_pib(self.current_date)

            # TODO: agent can split this QAP into FIL+, or RB, or a combination
            # how to decide??
            # if CC, then QA = RB, if FIL+, then RB = QA / filplus_multiplier

            # for now, we put all power into FIL+ (deal power)
            rb_to_onboard = min(max_possible_qa_power/constants.FIL_PLUS_MULTIPLER, self.model.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT)
            qa_to_onboard = apply_qa_multiplier(rb_to_onboard)

            # TODO: update to: put as much as possible into deal-power, and the remainder into CC power (renew first)
            self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, best_duration)

        super().step()