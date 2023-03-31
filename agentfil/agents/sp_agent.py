import mesa
from datetime import datetime, timedelta
import numpy as np
from numpy.random import default_rng
import pandas as pd
import os

from .. import constants
from ..power import cc_power, deal_power


class SPAgent(mesa.Agent):
    def __init__(self, model, agent_id, agent_seed, start_date, end_date, max_sealing_throughput_pib=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB):
        self.unique_id = agent_id  # the field unique_id is required by the framework
        self.model = model

        self.start_date = start_date  # used to get historical data

        # used for simulation purposes
        self.current_date = start_date
        self.end_date = end_date
        self.sim_len_days = (self.end_date - constants.NETWORK_DATA_START).days
        self.current_day = (self.current_date - constants.NETWORK_DATA_START).days

        self.max_sealing_throughput_pib = max_sealing_throughput_pib

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
        # self.accounting_df['capital_inflow_FIL'] = 0
        self.accounting_df['pledge_requested_FIL'] = 0
        self.accounting_df['pledge_repayment_FIL'] = 0
        self.accounting_df['pledge_interest_payment_FIL'] = 0

        self.agent_info_df = pd.DataFrame({'date': pd.date_range(start_date, end_date, freq='D')[:-1]})
        self.agent_info_df['cc_onboarded'] = 0
        self.agent_info_df['cc_renewed'] = 0
        self.agent_info_df['cc_onboarded_duration'] = 0
        self.agent_info_df['cc_renewed_duration'] = 0
        self.agent_info_df['deal_onboarded'] = 0
        self.agent_info_df['deal_renewed'] = 0
        self.agent_info_df['deal_onboarded_duration'] = 0
        self.agent_info_df['deal_renewed_duration'] = 0

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

    def onboard_power(self, date_in, cc_power_in_pib, qa_power_in_pib, duration_days, pledge_for_onboarding, pledge_repayment_amount):
        """
        A convenience function which onboards the specified amount of power for a given date
        and updates other necessary internal variables to keep the agent in sync

        NOTE: while the date_in argument makes this function general, it is intended to use
        the current date of the agent. An assessment as to how using this function with
        dates not in sequential order synced to the current date affects the network statistics
        has not yet been conducted!

        Returns 0 if the power was successfully onboarded, and -1 if the power was not
        """

        # enforce min/max onboarding power rules set forth by the network
        if cc_power_in_pib < self.model.MIN_DAY_ONBOARD_RBP_PIB_PER_AGENT:
            # if agent tries to onboard less than 1 sector, return
            return -1
        if cc_power_in_pib > self.max_sealing_throughput_pib:
            # if agent tries to onboard more than maximum allowable, return
            return -1

        # convert the date to the index of the day in the simulation
          # using filecoin_df to find the index works because the vectors self.onboarded_power, self.t, and self.model.filecoin_df are all aligned
        day_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]
        self.onboarded_power[day_idx][0] += cc_power(cc_power_in_pib, duration_days)
        self.onboarded_power[day_idx][1] += deal_power(qa_power_in_pib, duration_days)

        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(date_in)].index[0]
        self.agent_info_df.loc[agent_df_idx, 'cc_onboarded'] += cc_power_in_pib
        self.agent_info_df.loc[agent_df_idx, 'deal_onboarded'] += qa_power_in_pib
        
        # NOTE: we overwrite the sector duration here because agents are tracked in aggregate 
        # over each day, rather than by each sector. This means that an inherent assumption
        # of this model is that the duration of the sector is the same for all sectors on a given day.
        self.agent_info_df.loc[agent_df_idx, 'cc_onboarded_duration'] = duration_days 
        self.agent_info_df.loc[agent_df_idx, 'deal_onboarded_duration'] = duration_days

        self._account_pledge_repayment_FIL(date_in, duration_days, pledge_for_onboarding, pledge_repayment_amount)

        return 0
    
    def renew_power(self, date_in, cc_power_in_pib, duration_days, pledge_for_renewing, pledge_repayment_amount):
        day_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]
        self.renewed_power[day_idx][0] += cc_power(cc_power_in_pib, duration_days)
        
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(date_in)].index[0]
        self.agent_info_df.loc[agent_df_idx, 'cc_renewed'] += cc_power_in_pib
        self.agent_info_df.loc[agent_df_idx, 'cc_renewed_duration'] = duration_days

        # NOTE: Deal power cannot be renewed in the current spec. However, recall that CC power
        # gets a QA multiplier of 1, and thus it is part of the "QA" power in the model. The filecoin_model.py
        # module CC and QA power separately, so we include the CC power renewed in the QA power calculations
        # by adding it to the deal_power vector.
        self.renewed_power[day_idx][1] += deal_power(cc_power_in_pib, duration_days)
        self.agent_info_df.loc[agent_df_idx, 'deal_renewed'] += cc_power_in_pib
        self.agent_info_df.loc[agent_df_idx, 'deal_renewed_duration'] = duration_days

        self._account_pledge_repayment_FIL(date_in, duration_days, pledge_for_renewing, pledge_repayment_amount)

    def _account_pledge_repayment_FIL(self, date_in, sector_duration_days, pledge_requested_FIL=0, pledge_repayment_FIL=0):
        df_idx = self.accounting_df[self.accounting_df['date'] == date_in].index[0]
        self.accounting_df.loc[df_idx, 'pledge_requested_FIL'] += pledge_requested_FIL
        
        # account for repayment when sector ends
        df_idx += sector_duration_days
        if df_idx < len(self.accounting_df):
            self.accounting_df.loc[df_idx, 'pledge_repayment_FIL'] += pledge_repayment_FIL
            self.accounting_df.loc[df_idx, 'pledge_interest_payment_FIL'] += (pledge_repayment_FIL - pledge_requested_FIL)
            
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

    def get_se_power_at_date(self, date_in):
        power_at_date = self.get_power_at_date(date_in)
        return {
            'se_cc_power': power_at_date['sched_expire_rb_pib'],
            'se_deal_power': power_at_date['sched_expire_qa_pib'],
        }

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

    
    def _get_available_FIL(self, date_in):
        """
        Returns the amount of FIL available to the agent on the given date.
        In this implementation, we compute the total FIL available by summing rewards, pledges, and capital inflows.
        This makes sense if using the capital inflow model. If using the discount rate model, this function
        is not relevant in that context.
        """
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_in)].index[0]
        accounting_df_subset = self.accounting_df.loc[0:accounting_df_idx, :]
        
        available_FIL = np.sum(accounting_df_subset['reward_FIL']) \
                        - np.sum(accounting_df_subset['onboard_pledge_FIL']) \
                        - np.sum(accounting_df_subset['renew_pledge_FIL']) \
                        + np.sum(accounting_df_subset['onboard_scheduled_pledge_release_FIL']) \
                        + np.sum(accounting_df_subset['renew_scheduled_pledge_release_FIL']) \
                        + np.sum(accounting_df_subset['capital_inflow_FIL'])
        return available_FIL

    def get_available_FIL(self, date_in):
        if hasattr(self.model, 'capital_inflow_process'):
            return self._get_available_FIL(date_in)
        else:
            raise NotImplementedError("get_available_FIL not implemented when using discount rate model")
    
    def get_reward_FIL(self, date_in):
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_in)].index[0]
        accounting_df_subset = self.accounting_df.loc[0:accounting_df_idx, :]
        
        return np.sum(accounting_df_subset['reward_FIL'])
    
    def get_net_reward_FIL(self, date_in):
        """
        Returns rewards - (pledge delta due to borrowing costs)
        """
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_in)].index[0]
        accounting_df_subset = self.accounting_df.loc[0:accounting_df_idx, :]
        
        return np.sum(accounting_df_subset['reward_FIL']) - np.sum(accounting_df_subset['pledge_delta_FIL'])

    def get_max_onboarding_qap_pib(self, date_in):
        available_FIL = self.get_available_FIL(date_in)
        if available_FIL > 0:
            pledge_per_pib = self.model.estimate_pledge_for_qa_power(date_in, 1.0)
            pibs_to_onboard = available_FIL / pledge_per_pib
        else:
            pibs_to_onboard = 0
        if np.isnan(pibs_to_onboard):
            raise ValueError("Pibs to onboard yielded NAN")
        return pibs_to_onboard
    
    def compute_actual_power_possible_to_onboard_from_supply_discount_rate_model(self, desired_power_to_onboard, duration):
        pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)
        total_pledge_needed_for_onboards = desired_power_to_onboard * pledge_per_pib
        available_FIL = self.model.borrow_FIL_with_discount_rate(self.current_date, total_pledge_needed_for_onboards, duration)
        power_to_onboard = available_FIL / pledge_per_pib
        return power_to_onboard, available_FIL, total_pledge_needed_for_onboards
    
    def compute_repayment_amount_from_supply_discount_rate_model(self, date_in, pledge_amount, duration_yrs):
        # treat the pledge amount as the current value, and compute future value based on the discount rate
        discount_rate_pct = self.model.get_discount_rate_pct(date_in)
        discount_rate = discount_rate_pct / 100.0
        
        #future_value = pledge_amount * (1 + (discount_rate/compounding_freq_yrs)) ** (duration_yrs*compounding_freq_yrs)
        future_value = pledge_amount * np.exp(discount_rate * duration_yrs)
        return future_value

    def save_data(self, output_dir):
        accounting_fp = os.path.join(output_dir, 'agent_%d_accounting_info.csv' % (self.unique_id,))
        self.accounting_df.to_csv(accounting_fp, index=False)

        agent_info_fp = os.path.join(output_dir, 'agent_%d_agent_info.csv' % (self.unique_id,))
        self.agent_info_df.to_csv(agent_info_fp, index=False)