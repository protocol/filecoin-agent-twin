from datetime import timedelta, datetime
import time
import datetime as dt
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power
from .filecoin_model import apply_qa_multiplier

from scenario_generator.gbm_forecast import gbm_forecast

import copy
import numpy as np
import pandas as pd

from pycoingecko import CoinGeckoAPI

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
    duration_vec_days = [6, 12, 36, 60]*30
    profitability_vec = []
    for d in duration_vec_days:
        - Estimate ROI
            - x = Estimate future rewards / sector # currently linear extrapolation of last day of data for duration d
            - roi = (np.sum(x)/pledge_per_sector)
            - duration_yrs = d/360
            - roi_annualized = (1+roi)^(1/duration_yrs) - 1
        - Estimate fees for onboarding power
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
    [ ] - Iterative estimation of block rewards / day
    [ ] - How to model external macro-environmental factors (interest rates, etc.) in order to
          determine if agent should borrow money to onboard more power than it has reserves for?

    """

    def __init__(self, model, id, historical_power, start_date, end_date, 
                 accounting_df=None, random_seed=1111, agent_optimism=3,
                 forecast_num_mc=200):
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
            forecast_num_mc: the number of monte carlo simulations to run when forecasting anything in the agent
                some forecasting happens every time step, so be mindful of this parameter when running simulations
        """
        super().__init__(model, id, historical_power, start_date, end_date)
        
        self.random_seed = random_seed
        self.duration_vec_days = np.asarray([6, 12, 36])*30
        self.agent_optimism = agent_optimism
        self.forecast_num_mc = forecast_num_mc

        self.validate(accounting_df)
        self.map_optimism_scales()
        self.get_exchange_rate()
        self.forecast_future_exchange_rate()

        # good for debugging agent actions.  
        # consider paring it down when not debugging for simulation speed
        self.agent_info_df = copy.copy(accounting_df)
        self.agent_info_df['roi_estimate_6mo'] = 0
        self.agent_info_df['roi_estimate_1y'] = 0
        self.agent_info_df['roi_estimate_3y'] = 0
        self.agent_info_df['profit_duration_6mo'] = 0
        self.agent_info_df['profit_duration_1y'] = 0
        self.agent_info_df['profit_duration_3y'] = 0
        self.agent_info_df['cc_onboarded'] = 0
        self.agent_info_df['cc_renewed'] = 0
        self.agent_info_df['cc_onboarded_duration'] = 0
        self.agent_info_df['cc_renewed_duration'] = 0
        self.agent_info_df['deal_onboarded'] = 0
        self.agent_info_df['deal_renewed'] = 0
        self.agent_info_df['deal_onboarded_duration'] = 0
        self.agent_info_df['deal_renewed_duration'] = 0
        self.agent_info_df['USD_used'] = 0

    def map_optimism_scales(self):
        self.optimism_to_price_quantile = {
            1 : 0.1,
            2 : 0.25,
            3 : 0.5,
            4 : 0.75,
            5 : 0.90
        }
        self.optimism_to_dayrewardspersector_quantile = {
            1 : 0.1,
            2 : 0.25,
            3 : 0.5,
            4 : 0.75,
            5 : 0.90
        }
        

    def validate(self, accounting_df):
        if accounting_df is None:
            raise ValueError("The accounting_df must be specified.")
        # check the length of the usd_df and the bounds of the dates
        if accounting_df.shape[0] != (self.end_date - self.start_date).days:
            raise ValueError("The length of the usd_df must be the same as the simulation length.")
        if pd.to_datetime(accounting_df.iloc[0]['date']) != pd.to_datetime(self.start_date):
            raise ValueError("The first date in the usd_df must be the same as the start_date.")
        # if self.accounting_df.iloc[-1]['date'] != self.end_date:
        #     raise ValueError("The last date in the usd_df must be the same as the end_date.")
        
        if 'date' not in accounting_df.columns:
            raise ValueError("The usd_df must have a date column.")
        if 'USD' not in accounting_df.columns:
            raise ValueError("The usd_df must have a USD column.")

        assert self.agent_optimism >= 1 and self.agent_optimism <= 5, \
                "fil_usd_price_optimism_scale must be between 1 and 5"
        assert type(self.agent_optimism) == int, "fil_usd_price_optimism_scale must be an integer"

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

    def forecast_future_exchange_rate(self):
        # update the prediction to something better
        last_price = self.usd_fil_exchange_df.iloc[-1]['price']
        remaining_len = (self.end_date - self.start_date).days + 1

        # use Geometric Brownian Motion to predict future prices, market caps, and total volumes
        # TODO: run prediction on market-cap & volume, but this information is not currently 
        # used by this agent, so I left it out currently
        x = self.usd_fil_exchange_df['price'].values
        forecast_length = remaining_len + np.max(self.duration_vec_days)
        num_mc = self.forecast_num_mc
        seed_base = self.random_seed
        future_prices_vec = []
        for ii in range(num_mc):
            seed_in = seed_base + ii
            y = gbm_forecast(x, forecast_length, seed=seed_in)
            future_prices_vec.append(y)

        q = self.optimism_to_price_quantile[self.agent_optimism]
        future_price = np.quantile(np.asarray(future_prices_vec), q, axis=0)
        
        future_price_df = pd.DataFrame(
            {
                "coin" : 'filecoin',
                "date" : pd.date_range(self.start_date, periods=forecast_length, freq='D'),
                "price" : future_price,
                "market_caps" : np.random.normal(last_price, 0.5, forecast_length),
                "total_volumes" : np.random.normal(last_price, 0.5, forecast_length)
            }
        )
        future_price_df['date'] = pd.to_datetime(future_price_df['date']).dt.date
        self.usd_fil_exchange_df = pd.concat([self.usd_fil_exchange_df, future_price_df], ignore_index=True)


    def get_available_FIL(self, date_in):
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_in)].index[0]
        accounting_df_subset = self.accounting_df.loc[0:accounting_df_idx, :]
        available_FIL = accounting_df_subset['reward_FIL'].cumsum() \
                        - accounting_df_subset['onboard_pledge_FIL'].cumsum() \
                        - accounting_df_subset['renew_pledge_FIL'].cumsum() \
                        + accounting_df_subset['onboard_scheduled_pledge_release_FIL'].cumsum() \
                        + accounting_df_subset['renew_scheduled_pledge_release_FIL'].cumsum()
        return available_FIL.values[-1]

    def get_max_onboarding_qap_pib(self, date_in):
        available_FIL = self.get_available_FIL(date_in)
        pledge_per_pib = self.model.estimate_pledge_for_qa_power(date_in, 1.0)
        pibs_to_onboard = available_FIL / pledge_per_pib
        
        return pibs_to_onboard

    def forecast_day_rewards_per_sector(self, forecast_start_date, forecast_length):
        df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(forecast_start_date)].index[0]

        # # GBM method
        # x = self.model.filecoin_df.loc[0:df_idx-1, 'day_rewards_per_sector'].values
        # num_mc = self.forecast_num_mc
        # seed_base = self.random_seed
        # day_rewards_per_sector_forecast_vec = []
        # for ii in range(num_mc):
        #     seed_in = seed_base + ii
        #     y = gbm_forecast(x, forecast_length, seed=seed_in)
        #     day_rewards_per_sector_forecast_vec.append(y)
        # q = self.optimism_to_dayrewardspersector_quantile[self.agent_optimism]
        # future_rewards_per_sector = np.quantile(np.asarray(day_rewards_per_sector_forecast_vec), q, axis=0)
        # return future_rewards_per_sector

        # lazy method
        future_rewards_per_sector = np.ones(forecast_length) * self.model.filecoin_df.loc[df_idx-1, 'day_rewards_per_sector']
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
        
        return roi_estimate_annualized


    def step(self):
        roi_estimate_vec = []
        profitability_vec = []

        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(self.current_date)].index[0]
        prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']
        current_exchange_rate = self.usd_fil_exchange_df.loc[self.usd_fil_exchange_df['date'] == self.current_date, 'price'].values[0]
        for d in self.duration_vec_days:    
            roi_estimate = self.estimate_roi(d, self.current_date)
            
            # Estimate future USD/FIL exchange rate at time t+d
            future_date = self.current_date + timedelta(days=int(d))
            future_exchange_rate = self.usd_fil_exchange_df.loc[self.usd_fil_exchange_df['date'] == future_date, 'price'].values[0]
        
            profit_metric = (1+roi_estimate)*future_exchange_rate - current_exchange_rate
            
            roi_estimate_vec.append(roi_estimate)
            profitability_vec.append(profit_metric)
        
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(self.current_date)].index[0]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_6mo'] = roi_estimate_vec[0]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_1y'] = roi_estimate_vec[1]
        self.agent_info_df.loc[agent_df_idx, 'roi_estimate_3y'] = roi_estimate_vec[2]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_6mo'] = profitability_vec[0]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_1y'] = profitability_vec[1]
        self.agent_info_df.loc[agent_df_idx, 'profit_duration_3y'] = profitability_vec[2]

        max_profit_idx = np.argmax(profitability_vec)
        best_duration = self.duration_vec_days[max_profit_idx]
        if profitability_vec[max_profit_idx] > 0: 
            max_possible_qa_power = self.get_max_onboarding_qap_pib(self.current_date)

            # TODO: agent can split this QAP into FIL+, or RB, or a combination
            # how to decide??
            # if CC, then QA = RB, if FIL+, then RB = QA / filplus_multiplier

            # for now, we put all power into FIL+ (deal power)
            rb_to_onboard = min(max_possible_qa_power/constants.FIL_PLUS_MULTIPLER, constants.MAX_DAY_ONBOARD_RBP_PIB)
            qa_to_onboard = apply_qa_multiplier(rb_to_onboard)

            self.onboarded_power[self.current_day][0] += cc_power(rb_to_onboard, best_duration)
            self.onboarded_power[self.current_day][1] += deal_power(qa_to_onboard, best_duration)

            # TODO: update to: put as much as possible into deal-power, and the remainder into CC power (renew first)

            # bookkeeping to track/debug agents
            self.agent_info_df.loc[agent_df_idx, 'cc_onboarded'] = rb_to_onboard
            self.agent_info_df.loc[agent_df_idx, 'cc_onboarded_duration'] = best_duration
            self.agent_info_df.loc[agent_df_idx, 'deal_onboarded'] = qa_to_onboard
            self.agent_info_df.loc[agent_df_idx, 'deal_onboarded_duration'] = best_duration

        # update when the onboarded power is scheduled to expire
        super().step()

    def post_global_step(self):
        # we can update local representation of anything else that should happen after
        # global metrics for day are aggregated
        pass