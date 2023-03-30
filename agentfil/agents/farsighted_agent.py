from datetime import timedelta
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power
from .filecoin_model import apply_qa_multiplier

import numpy as np
import pandas as pd

class FarSightedAgent(SPAgent):
    """
    The FarSightedAgent is a one-step evolution of the Greedy Agent. At a high level,
    it works by trying to predict the ROI and profit metric for onboarding power over a given duration,
    indicated by the far_sightedness_days parameter for possible sector durations. Define W
    to be the window of time over which the agent is trying to predict its ROI.
    In W, the agent will then use a "max-filling" strategy to onboard power. This means that
    in W, it will onboard as much power as it can for the maximum profit day, and then it will
    repeat this process going down the list of profitable days in descending order until it either 
    runs out of FIL or the end of W is reached. 
    
    The agent can be configured to re-estimate ROI & profit over the defined farsightedness duration 
    at a given cadence. If re-estimation is done every day, note that this would be computationally intensive
    and may not be feasible for large simulations.

    Additionally, note that re-estimation at a faster rate than the rewards_per_sector_process in the main
    is ineffective because this agent uses the predictions made by the rewards_per_sector_process to estimate ROI.
    However, if the agent were making its own predictions in addition to that, that were relevant
    to its decision making at a faster cadence, then it may make sense to re-estimate at a faster rate.
    """
    def __init__(self, model, id, historical_power, start_date, end_date, 
                 random_seed=1111, 
                 agent_optimism=3, 
                 far_sightedness_days=90, reestimate_every_days=90):
        super().__init__(model, id, historical_power, start_date, end_date)

        self.random_seed = random_seed
        self.duration_vec_days = (np.asarray([12, 36])*30).astype(np.int32)
        self.agent_optimism = agent_optimism
        self.far_sightedness_days = far_sightedness_days
        self.reestimate_every_days = reestimate_every_days

        start_date = self.model.start_date
        end_date = self.model.end_date
        self.reestimate_dates = [start_date + timedelta(days=i) for i in range(0, self.model.sim_len, reestimate_every_days)]

        self.map_optimism_scales()

        # this data-frame tracks the ROI estimates the agent is making 
        for d in self.duration_vec_days:
            self.agent_info_df[f'roi_estimate_{d}'] = 0
            self.agent_info_df[f'profit_metric_{d}'] = 0
        self.agent_info_df['scheduled_qa_power_pib'] = 0.
        self.agent_info_df['scheduled_rb_power_pib'] = 0.
        self.agent_info_df['scheduled_power_duration'] = 0.

        # don't need to generate this in a loop every time ... 
        self.profit_metric_cols = [x for x in self.agent_info_df.columns if 'profit_metric' in x]
        # print(self.agent_info_df[self.profit_metric_cols].dtypes)

        self.validate()

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
        if self.reestimate_every_days > self.far_sightedness_days:
            raise ValueError("The re_estimate_every_days must be less than or equal to far_sightedness_days")
        if self.reestimate_every_days < 1:
            raise ValueError("The re_estimate_every_days must be greater than or equal to 1")
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
        # we use the current date-1 when accessing the day_pledge_per_qap because that is the most 
        # upto date estimate we have of day_pledge_per_QAP
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(self.current_date)].index[0]
        prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']

        # TODO: make this an iterative update rather than full-estimate every day
        future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        roi_estimate = future_rewards_per_sector_estimate.sum() / prev_day_pledge_per_QAP
        
        # annualize it so that we can have the same frame of reference when comparing different sector durations
        duration_yr = sector_duration / 360.0  
        roi_estimate_annualized = (1.0+roi_estimate)**(1.0/duration_yr) - 1
        
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
        ### called every tick
        if self.model.current_date in self.reestimate_dates:
            # plan out the next "far_sightedness_days" window of power scheduling

            estimation_day = self.model.current_date
            current_exchange_rate = self.get_exchange_rate(self.current_date)
            print("Estimating ROI over window ...")
            for ii in range(self.far_sightedness_days):
                # estimate the ROI for each sector duration
                for sector_duration in self.duration_vec_days:
                    roi_estimate = self.estimate_roi(sector_duration, estimation_day)
                    # Estimate future USD/FIL exchange rate at time t+d
                    future_date = estimation_day + timedelta(days=int(sector_duration))
                    future_exchange_rate = self.get_exchange_rate(future_date)
                
                    profit_metric = (1+roi_estimate)*future_exchange_rate - current_exchange_rate
                    
                    self.agent_info_df.loc[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(estimation_day), 
                                           'roi_estimate_%d' % (sector_duration,)] = float(roi_estimate)
                    self.agent_info_df.loc[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(estimation_day), 
                                           'profit_metric_%d' % (sector_duration,)] = float(profit_metric)

                estimation_day += timedelta(days=1)

            max_possible_qa_power_pib = self.get_max_onboarding_qap_pib(self.current_date)
            max_profit_by_day = self.agent_info_df[self.profit_metric_cols].max(axis=1)
            power_filling_order = np.argsort(max_profit_by_day.values)[::-1]  # get days to fill power in descending order of profit
            for ii in range(self.far_sightedness_days):
                day_idx = power_filling_order[ii]
                if max_profit_by_day.iloc[day_idx] > 0:
                    # determine the best duration
                    # this is really shaky and error prone with the string parsing, but it works for now
                    # revisit
                    best_duration = int(self.agent_info_df.iloc[day_idx][self.profit_metric_cols].astype(float).idxmax().split('_')[-1])
                    
                    # schedule the maximum possible power to be onboarded on this day (it is all FIL+ power for now)
                    rb_to_onboard = min(max_possible_qa_power_pib/constants.FIL_PLUS_MULTIPLER, self.model.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT)
                    qa_to_onboard = apply_qa_multiplier(rb_to_onboard)

                    self.agent_info_df.loc[day_idx, 'scheduled_qa_power_pib'] = qa_to_onboard
                    self.agent_info_df.loc[day_idx, 'scheduled_rb_power_pib'] = rb_to_onboard
                    self.agent_info_df.loc[day_idx, 'scheduled_power_duration'] = best_duration

                    max_possible_qa_power_pib -= qa_to_onboard


        # add power for the day according to the generated schedule
        agent_info_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(self.current_date)].index[0]
        qa_to_onboard = self.agent_info_df.loc[agent_info_df_idx, 'scheduled_qa_power_pib']
        rb_to_onboard = self.agent_info_df.loc[agent_info_df_idx, 'scheduled_rb_power_pib']
        duration = self.agent_info_df.loc[agent_info_df_idx, 'scheduled_power_duration']

        self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, duration)

        super().step()