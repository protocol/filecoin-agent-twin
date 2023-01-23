from datetime import timedelta, datetime
import time
import datetime as dt
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power

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
    """

    def __init__(self, model, id, historical_power, start_date, end_date, accounting_df=None):
        super().__init__(model, id, historical_power, start_date, end_date)
        
        self.duration_vec = [6, 12, 36, 60]
        self.accounting_df = accounting_df

        self.FIL_USD_EXCHANGE_EFFICIENCY = 0.7
        self.ROI_EFFICIENCY_FACTOR = 0.7  # a simple model to account for gas fees and opex

        self.validate()
        self.get_exchange_rate()

    def validate(self):
        if self.accounting_df is None:
            raise ValueError("The accounting_df must be specified.")
        # check the length of the usd_df and the bounds of the dates
        if self.accounting_df.shape[0] != (self.end_date - self.start_date).days:
            raise ValueError("The length of the usd_df must be the same as the simulation length.")
        if self.accounting_df.iloc[0]['date'] != self.start_date:
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
                                                  to_timestamp=time.mktime(self.start_date.timetuple()))

        self.usd_fil_exchange_df = pd.DataFrame(
            {
                "coin" : id_,
                "time_s" : np.array(ts['prices']).T[0],
                "time_d" : list(map(change_t, np.array(ts['prices']).T[0])),
                "price" : np.array(ts['prices']).T[1],
                "market_caps" : np.array(ts['market_caps']).T[1], 
                "total_volumes" : np.array(ts['total_volumes']).T[1]
            }
        )
        self.usd_fil_exchange_df['time_d'] = pd.to_datetime(self.usd_fil_exchange_df['time_d'])
                                             

    def get_max_onboarding_power(self, date_in):
        accounting_df_idx = self.accounting_df[self.accounting_df['date'] == date_in].index[0]
        available_USD = self.accounting_df.loc[accounting_df_idx, 'USD'] - self.accounting_df.loc[0:accounting_df_idx, 'funds_used'].sum()
        # TODO: need to address what happens if we don't have the exchange rate for the day (either data is missing or in the future)
        current_exchange_rate = self.usd_fil_exchange_df.loc[self.usd_fil_exchange_df['time_d'] == date_in, 'price'].values[0]
        available_FIL = available_USD / current_exchange_rate
        available_FIL_after_fees = available_FIL * self.FIL_USD_EXCHANGE_EFFICIENCY

        sectors_available_to_onboard = available_FIL_after_fees / self.model.filecoin_df.loc[date_in, 'day_pledge_per_QAP']
        return sectors_available_to_onboard * constants.SECTOR_SIZE


    def step(self):
        profitability_vec = []
        # TODO: take into account duration in all calculations. currently we use random noise to simulate this
        for d in self.duration_vec:
            # Estimate Instantaneous ROI and use that as future ROI
            roi_estimate = self.model.filecoin_df.loc[self.model.current_day, 'day_rewards_per_sector'] / self.model.filecoin_df.loc[self.model.current_day, 'day_pledge_per_QAP']
            
            # add in a factor to account for interest rates, fees, etc.
            roi_estimate = roi_estimate * self.ROI_EFFICIENCY_FACTOR
            
            current_exchange_rate = self.usd_fil_exchange_df.loc[self.usd_fil_exchange_df['time_d'] == self.current_date, 'price'].values[0]
            # Estimate future USD/FIL exchange rate at time t+d
            future_exchange_rate = current_exchange_rate + np.random.normal(0, 0.5)
        
            profit_metric = (1+roi_estimate)*future_exchange_rate - current_exchange_rate
            profitability_vec.append(profit_metric)

        max_profit_idx = np.argmax(profitability_vec)
        best_duration = self.duration_vec[max_profit_idx]
        if profitability_vec[max_profit_idx] > 0:
            # check how much power we can onboard, this is based the amount of money we have
            max_onboarding_power = self.get_max_onboarding_power(self.current_date)

            # of the maximum available to onboard, choose a certain amount
            power_to_onboard = max_onboarding_power * np.random.beta(2, 2)

            # of the power to onboard, split evenly between CC and Deals.
            cc_total = power_to_onboard / 2.
            # onboard deal power only b/c we can't renew deal power
            deal_to_onboard = power_to_onboard / 2.
            self.onboarded_power[self.current_day][1] += deal_power(deal_to_onboard, best_duration)
            
            # for CC power, renew as much as possible first and onboard the remainder
            today_cc_expire = self.scheduled_expire_power[self.current_day][0]
            cc_renew = min(today_cc_expire, cc_total)
            cc_onboard = cc_total - cc_renew
            if cc_renew > 0:
                # renew all of the CC power
                self.renewed_power[self.current_day][0] += cc_power(cc_renew, best_duration)
            if cc_onboard > 0:
                self.onboarded_power[self.current_day][0] += cc_power(cc_onboard, best_duration)

            # update local representation of available funds
            funds_used = power_to_onboard * self.model.filecoin_df.loc[self.current_date, 'day_pledge_per_QAP'] * current_exchange_rate
            self.accounting_df.loc[self.current_date, 'funds_used'] += funds_used

        # update when the onboarded power is scheduled to expire
        super().step()