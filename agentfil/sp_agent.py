import mesa
from datetime import datetime, timedelta
import numpy as np
from numpy.random import default_rng
import pandas as pd

from . import constants
from .power import cc_power, deal_power


class SPAgent(mesa.Agent):
    def __init__(self, model, agent_id, agent_seed, start_date, end_date):
        self.unique_id = agent_id  # the field unique_id is required by the framework
        self.model = model

        self.start_date = start_date  # used to get historical data

        # used for simulation purposes
        self.current_date = start_date
        self.end_date = end_date
        self.sim_len_days = (self.end_date - constants.NETWORK_DATA_START).days
        self.current_day = (self.current_date - constants.NETWORK_DATA_START).days

        ########################################################################################
        # NOTE: self.t should be equal to self.model.filecoin_df['date']
        self.t = [start_date + timedelta(days=i) for i in range(self.sim_len_days)]
        ########################################################################################

        self.onboarded_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        self.renewed_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        self.terminated_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]

        # NOTE: duration is not relevant for SE power. It is only relevant for onboarded or renewed power
        self.scheduled_expire_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]

        self.accounting_df = pd.DataFrame()
        self.accounting_df['date'] = self.model.filecoin_df['date']
        self.accounting_df['reward_FIL'] = 0
        self.accounting_df['onboard_pledge_FIL'] = 0
        self.accounting_df['renew_pledge_FIL'] = 0
        self.accounting_df['onboard_scheduled_pledge_release_FIL'] = 0
        self.accounting_df['renew_scheduled_pledge_release_FIL'] = 0
        self.accounting_df['capital_inflow_FIL'] = 0

        self.allocate_historical_power(agent_seed)

    def step(self):
        """
        Make a decision to onboard new power, renew or terminate existing power, or a combination
        based on a utility function

        # NOTE: when we add new power, we need to commensurately update:
        #  1 - the scheduled-expire-power based on duration
        #  2 - the scheduled-expire-pledge based on duration?? (or is it 180 days flat?)

        WARNING: this implementation of step does common things that should be done by any agent.
        In order to keep implementations clean and streamlined, 
        """
        self._bookkeep()

    def post_global_step(self):
        pass

    def _bookkeep(self):
        # book-keeping stuff that is common for any type of agent
        # automatically update the SE power based on what was onboarded & renewed
        # update internal representation of day/date

        today_onboarded_cc_power = self.onboarded_power[self.current_day][0]
        today_onboarded_deal_power = self.onboarded_power[self.current_day][1]

        cc_expire_index = self.current_day + today_onboarded_cc_power.duration
        deal_expire_index = self.current_day + today_onboarded_deal_power.duration

        if cc_expire_index < self.sim_len_days:
            self.scheduled_expire_power[cc_expire_index][0] += today_onboarded_cc_power
        if deal_expire_index < self.sim_len_days:
            self.scheduled_expire_power[deal_expire_index][1] += today_onboarded_deal_power

        # do the same for renewals
        today_renewed_cc_power = self.renewed_power[self.current_day][0]
        today_renewed_deal_power = self.renewed_power[self.current_day][1]

        cc_expire_index = self.current_day + today_onboarded_cc_power.duration
        deal_expire_index = self.current_day + today_onboarded_deal_power.duration

        if cc_expire_index < self.sim_len_days:
            self.scheduled_expire_power[cc_expire_index][0] += today_renewed_cc_power
        if deal_expire_index < self.sim_len_days:
            self.scheduled_expire_power[deal_expire_index][1] += today_renewed_deal_power

        self.current_day += 1
        self.current_date += timedelta(days=1)

    def disburse_rewards(self, date_in, reward):
        df_idx = self.accounting_df[self.accounting_df['date'] == date_in].index[0]
        self.accounting_df.loc[df_idx, 'reward_FIL'] += reward

    def get_power_at_date(self, date_in):
        ii = (date_in - constants.NETWORK_DATA_START).days
        assert ii >= 0, "date_in must be >= %s" % (constants.NETWORK_DATA_START,)
        out_dict = {
            'day_onboarded_rb_power_pib': self.onboarded_power[ii][0].pib,
            'day_onboarded_qa_power_pib': self.onboarded_power[ii][1].pib,
            'extended_rb_pib': self.renewed_power[ii][0].pib,
            'extended_qa_pib': self.renewed_power[ii][1].pib,
            'day_onboarded_qa_duration': self.onboarded_power[ii][1].duration,
            'extended_qa_duration': self.renewed_power[ii][1].duration,
            'sched_expire_rb_pib': self.scheduled_expire_power[ii][0].pib,
            'sched_expire_qa_pib': self.scheduled_expire_power[ii][1].pib,
            'terminated_rb_pib': self.terminated_power[ii][0].pib,
            'terminated_qa_pib': self.terminated_power[ii][1].pib,
            # 'scheduled_expire_pledge': self.scheduled_expire_pledge[ii],
        }
        return out_dict

    def allocate_historical_power(self, agent_seed):
        # need to distribute the power in the historical_power information from
        # day=0 to day=self.current_day-1
        historical_df = agent_seed['historical_power']
        future_se_df = agent_seed['future_se_power']

        # we can't vector assign b/c the power vectors are lists of objects, but 
        # this is premature optimization that we can revisit later
        global_ii = (historical_df.iloc[0]['date'] - constants.NETWORK_DATA_START).days
        ii_start = global_ii
        for _, row in historical_df.iterrows():
            self.onboarded_power[global_ii] = [
                cc_power(row['day_onboarded_rb_power_pib']),
                deal_power(row['day_onboarded_qa_power_pib'])
            ]
            self.renewed_power[global_ii] = [
                cc_power(row['extended_rb_pib']),
                deal_power(row['extended_qa_pib'])
            ]
            self.scheduled_expire_power[global_ii] = [
                cc_power(row['sched_expire_rb_pib']),
                deal_power(row['sched_expire_qa_pib'])
            ]
            self.terminated_power[global_ii] = [
                cc_power(row['terminated_rb_pib']),
                deal_power(row['terminated_qa_pib'])
            ]
            # self.scheduled_expire_pledge[global_ii] = row['total_pledge']
            
            global_ii += 1
        # print("Seeding agent: %d from index=%d:%d" % (self.unique_id, ii_start, global_ii))

        # add in the SE power
        global_ii = (future_se_df.iloc[0]['date'] - constants.NETWORK_DATA_START).days
        for _, row in future_se_df.iterrows():
            self.scheduled_expire_power[global_ii] = [
                cc_power(row['sched_expire_rb_pib']),
                deal_power(row['sched_expire_qa_pib'])
            ]
            # self.scheduled_expire_pledge[global_ii] = row['total_pledge']

            global_ii += 1

    def get_available_FIL(self, date_in):
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_in)].index[0]
        accounting_df_subset = self.accounting_df.loc[0:accounting_df_idx, :]
        
        available_FIL = np.sum(accounting_df_subset['reward_FIL']) \
                        - np.sum(accounting_df_subset['onboard_pledge_FIL']) \
                        - np.sum(accounting_df_subset['renew_pledge_FIL']) \
                        + np.sum(accounting_df_subset['onboard_scheduled_pledge_release_FIL']) \
                        + np.sum(accounting_df_subset['renew_scheduled_pledge_release_FIL']) \
                        + np.sum(accounting_df_subset['capital_inflow_FIL'])
        return available_FIL