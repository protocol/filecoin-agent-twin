from datetime import timedelta, datetime
import time
import datetime as dt
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power

import copy
import numpy as np
import pandas as pd

from pycoingecko import CoinGeckoAPI

class GreedyAgent(SPAgent):
    """
    The agent has the following properties:
        - Starting capital (USD)
        - Capital inflow. This must be a dataframe with two columns, date and USD. The length
          must be the same as the simulation length. It represents the amount of USD 
          that is inflowed into the Agent for the given day.

    The Greedy agent operations with the following behavior:

    duration_vec = [6, 12, 36, 60]
    For each tick (day) denoted by t, the agent will:
        profitability_vec = []
        for d in duration_vec:
            - Estimate ROI
                - Includes fees
                - Includes USD lending rates
            - Estimate fees for onboarding power
            - Estimate future USD/FIL exchange rate at time t+d
            - profit_metric = (1+ROI)*exchange_rate[t+d] - exchange_rate[t]
            - profitability_vec.append(profit_metric)
        - max_profit_idx = argmax(profitability_vec)
        - best duration = duration_vec[max_profit_idx]
        if profitability_vec[max_profit_idx] > 0:
            - Onboard power for duration = best duration
            - Add power to scheduled_expire_power
            - Add power to renewed_power
        else:
            - Do nothing

    This is locally optimal behavior but does not consider how ROI may change
    in order to be more strategic about when to onboard power.

    Questions:
    1) Do our current metrics distinguish between Deal/CC? I don’t see how they do, but I think we need a metric of that sort to determine which kind of 
    2) How to determine onboard vs. renew?
        Perhaps we decide we have total X power we’d like to put forward at time=T and we split that between renewal & onboard.
        Decision may be easier if we have metrics to distinguish between CC & Deal, b/c only CC can be renewed.

    """

    def __init__(self, model, id, historical_power, start_date, end_date, accounting_df=None):
        super().__init__(model, id, historical_power, start_date, end_date)
        
        self.duration_vec = [6, 12, 36, 60]
        self.accounting_df = accounting_df

        self.FIL_USD_EXCHANGE_EFFICIENCY = 0.7
        self.ROI_EFFICIENCY_FACTOR = 0.7  # a simple model to account for gas fees and opex

        self.validate()
        self.get_exchange_rate()
        self.predict_future_exchange_rate()

        # good for debugging agent actions.  
        # consider paring it down when not debugging for simulation speed
        self.agent_info_df = copy.copy(self.accounting_df)
        self.agent_info_df['roi_estimate_6mo'] = 0
        self.agent_info_df['roi_estimate_1y'] = 0
        self.agent_info_df['roi_estimate_3y'] = 0
        self.agent_info_df['roi_estimate_5y'] = 0
        self.agent_info_df['profit_duration_6mo'] = 0
        self.agent_info_df['profit_duration_1y'] = 0
        self.agent_info_df['profit_duration_3y'] = 0
        self.agent_info_df['profit_duration_5y'] = 0
        self.agent_info_df['cc_onboarded'] = 0
        self.agent_info_df['cc_renewed'] = 0
        self.agent_info_df['cc_onboarded_duration'] = 0
        self.agent_info_df['cc_renewed_duration'] = 0
        self.agent_info_df['deal_onboarded'] = 0
        self.agent_info_df['deal_renewed'] = 0
        self.agent_info_df['deal_onboarded_duration'] = 0
        self.agent_info_df['deal_renewed_duration'] = 0


    def validate(self):
        if self.accounting_df is None:
            raise ValueError("The accounting_df must be specified.")
        # check the length of the usd_df and the bounds of the dates
        if self.accounting_df.shape[0] != (self.end_date - self.start_date).days:
            raise ValueError("The length of the usd_df must be the same as the simulation length.")
        if pd.to_datetime(self.accounting_df.iloc[0]['date']) != pd.to_datetime(self.start_date):
            raise ValueError("The first date in the usd_df must be the same as the start_date.")
        # if self.accounting_df.iloc[-1]['date'] != self.end_date:
        #     raise ValueError("The last date in the usd_df must be the same as the end_date.")
        
        if 'date' not in self.accounting_df.columns:
            raise ValueError("The usd_df must have a date column.")
        if 'USD' not in self.accounting_df.columns:
            raise ValueError("The usd_df must have a USD column.")
        self.accounting_df['funds_used'] = 0

    def get_exchange_rate(self, id_='filecoin'):
        cg = CoinGeckoAPI()
        change_t = lambda x : datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d')
        ts = cg.get_coin_market_chart_range_by_id(id=id_,
                                                  vs_currency='usd',
                                                  from_timestamp=time.mktime(constants.NETWORK_DATA_START.timetuple()),
                                                  to_timestamp=time.mktime((self.start_date-timedelta(days=1)).timetuple()))

        self.usd_fil_exchange_df = pd.DataFrame(
            {
                "coin" : id_,
                "date" : list(map(change_t, np.array(ts['prices']).T[0])),
                "price" : np.array(ts['prices']).T[1],
                "market_caps" : np.array(ts['market_caps']).T[1], 
                "total_volumes" : np.array(ts['total_volumes']).T[1]
            }
        )
        self.usd_fil_exchange_df['date'] = pd.to_datetime(self.usd_fil_exchange_df['date']).dt.date

    def predict_future_exchange_rate(self):
        # update the prediction to something better
        last_price = self.usd_fil_exchange_df.iloc[-1]['price']
        remaining_len = (self.end_date - self.start_date).days + 1
        future_price_df = pd.DataFrame(
            {
                "coin" : 'filecoin',
                "date" : pd.date_range(self.start_date, self.end_date, freq='D'),
                # "price" : np.random.normal(last_price, 0.5, remaining_len),
                "price" : np.ones(remaining_len)*last_price,
                "market_caps" : np.random.normal(last_price, 0.5, remaining_len),
                "total_volumes" : np.random.normal(last_price, 0.5, remaining_len)
            }
        )
        future_price_df['date'] = pd.to_datetime(future_price_df['date']).dt.date
        self.usd_fil_exchange_df = pd.concat([self.usd_fil_exchange_df, future_price_df], ignore_index=True)
                                             

    def get_max_onboarding_power_pib(self, date_in):
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_in)].index[0]
        available_USD = self.accounting_df.loc[accounting_df_idx, 'USD'] - self.accounting_df.loc[0:accounting_df_idx, 'funds_used'].sum()
        
        current_exchange_rate = self.usd_fil_exchange_df.loc[pd.to_datetime(self.usd_fil_exchange_df['date']) == pd.to_datetime(date_in), 'price'].values[0]
        available_FIL = available_USD / current_exchange_rate
        available_FIL_after_fees = available_FIL * self.FIL_USD_EXCHANGE_EFFICIENCY

        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]
        # NOTE: we need to use previous day b/c the full day's metrics haven't been aggregated yet
        sectors_available_to_onboard = available_FIL_after_fees / self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']
        return sectors_available_to_onboard * constants.SECTOR_SIZE / constants.PIB


    def step(self):
        roi_estimate_vec = []
        profitability_vec = []
        # TODO: take into account duration in all calculations. currently we use random noise to simulate this
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(self.current_date)].index[0]
        for d in self.duration_vec:
            # Estimate Instantaneous ROI and use that as future ROI
            # NOTE: we need to use yesterday's metrics b/c today's haven't yet been aggregated by the system yet
            roi_estimate = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_rewards_per_sector'] / self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']
            
            # add in a factor to account for interest rates, fees, etc.
            roi_estimate = roi_estimate * self.ROI_EFFICIENCY_FACTOR
            
            current_exchange_rate = self.usd_fil_exchange_df.loc[self.usd_fil_exchange_df['date'] == self.current_date, 'price'].values[0]
            # Estimate future USD/FIL exchange rate at time t+d
            future_date = self.current_date + timedelta(days=d)
            future_exchange_rate = self.usd_fil_exchange_df.loc[self.usd_fil_exchange_df['date'] == future_date, 'price'].values
            if len(future_exchange_rate) == 0:
                # if we don't have an exchange rate for the future, use the last value
                future_exchange_rate = self.usd_fil_exchange_df.iloc[-1]['price']
            else:
                future_exchange_rate = future_exchange_rate[0]
        
            profit_metric = (1+roi_estimate)*future_exchange_rate - current_exchange_rate
            
            roi_estimate_vec.append(roi_estimate)
            profitability_vec.append(profit_metric)
        
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(self.current_date)].index[0]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_6mo'] = roi_estimate_vec[0]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_1y'] = roi_estimate_vec[1]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_3y'] = roi_estimate_vec[2]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_5y'] = roi_estimate_vec[3]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_6mo'] = profitability_vec[0]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_1y'] = profitability_vec[1]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_3y'] = profitability_vec[2]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_5y'] = profitability_vec[3]

        max_profit_idx = np.argmax(profitability_vec)
        best_duration = self.duration_vec[max_profit_idx]
        if profitability_vec[max_profit_idx] > 0:
            # check how much power we can onboard, this is based the amount of money we have
            # TODO: need to update this b/c deal power is 10x more pledge but 10x more reward later.
            max_possible_power = self.get_max_onboarding_power_pib(self.current_date)

            # of the maximum available to onboard, choose a certain amount
            power_to_onboard = max_possible_power * np.random.beta(2, 2)

            # put all that into deal power
            # TODO: I think I need to apply the quality multiplier here
            self.onboarded_power[self.current_day][1] += deal_power(power_to_onboard, best_duration)

            # update to: put as much as possible into deal-power, and the remainder into CC power (renew first)

            # update local representation of available funds
            funds_used = power_to_onboard * self.model.filecoin_df.loc[self.current_day, 'day_pledge_per_QAP'] * current_exchange_rate
            self.accounting_df.loc[agent_df_idx, 'funds_used'] += funds_used

            # bookkeeping to debug agents later
            self.agent_info_df.loc[agent_df_idx, 'deal_onboarded'] = power_to_onboard
            self.agent_info_df.loc[agent_df_idx, 'deal_onboarded_duration'] = best_duration
            self.agent_info_df.loc[agent_df_idx, 'funds_used'] += funds_used

        # update when the onboarded power is scheduled to expire
        super().step()