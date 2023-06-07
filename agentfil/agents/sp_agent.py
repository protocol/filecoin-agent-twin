import mesa
from datetime import datetime, timedelta, date
import numpy as np
from numpy.random import default_rng
import pandas as pd
import os

from .. import constants
from ..power import cc_power, deal_power


class SPAgent(mesa.Agent):
    def __init__(self, model, agent_id, agent_seed, start_date, end_date, 
                 max_sealing_throughput_pib=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB):
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
        self.t = np.asarray([constants.NETWORK_DATA_START + timedelta(days=i) for i in range(self.sim_len_days)])
        ########################################################################################

        # do we need to parallel account fro this in onboarded_power and teh data frame??
        self.onboarded_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        self.renewed_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        self.terminated_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        # NOTE: duration is not relevant for SE power. It is only relevant for onboarded or renewed power
        self.scheduled_expire_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        self.known_scheduled_expire_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]
        self.modeled_scheduled_expire_power = [[cc_power(0), deal_power(0)] for _ in range(self.sim_len_days)]

        self.accounting_df = pd.DataFrame()
        self.accounting_df['date'] = self.model.filecoin_df['date']
        self.accounting_df['reward_FIL'] = 0
        self.accounting_df['full_reward_for_power_FIL'] = 0  # tracks the total amount of expected reward for onboarding power
                                                             # this is the full amount, not the vested version which is what is used
                                                             # to actually compute the balance.  Useful when processing terminations
                                                             # to delete any rewards that would have been vested
        self.accounting_df['onboard_pledge_FIL'] = 0
        self.accounting_df['renew_pledge_FIL'] = 0
        self.accounting_df['onboard_scheduled_pledge_release_FIL'] = 0
        self.accounting_df['renew_scheduled_pledge_release_FIL'] = 0
        self.accounting_df['scheduled_pledge_release'] = 0
        # self.accounting_df['capital_inflow_FIL'] = 0
        self.accounting_df['pledge_requested_FIL'] = 0
        self.accounting_df['pledge_repayment_FIL'] = 0
        self.accounting_df['pledge_interest_payment_FIL'] = 0
        self.accounting_df['termination_burned_FIL'] = 0

        # start one day before simulation start so that we can easily account for historical data without special logic
        # self.agent_info_df = pd.DataFrame({'date': pd.date_range(start_date-timedelta(days=1), end_date, freq='D')[:-1]})
        self.agent_info_df = pd.DataFrame({'date': pd.date_range(constants.NETWORK_DATA_START, end_date, freq='D')[:-1]})
        # TODO: I think the non cumulative columns are not needed
        self.agent_info_df['cc_onboarded'] = 0
        self.agent_info_df['cc_renewed'] = 0
        self.agent_info_df['cc_onboarded_duration'] = 0
        self.agent_info_df['cc_renewed_duration'] = 0
        self.agent_info_df['cum_cc_onboarded'] = 0
        self.agent_info_df['cum_cc_renewed'] = 0
        self.agent_info_df['cum_cc_sched_expire'] = 0
        self.agent_info_df['cum_cc_terminated'] = 0

        self.agent_info_df['deal_onboarded'] = 0
        self.agent_info_df['deal_renewed'] = 0
        self.agent_info_df['deal_onboarded_duration'] = 0
        self.agent_info_df['deal_renewed_duration'] = 0
        self.agent_info_df['cum_deal_onboarded'] = 0
        self.agent_info_df['cum_deal_renewed'] = 0
        self.agent_info_df['cum_deal_sched_expire'] = 0
        self.agent_info_df['cum_deal_terminated'] = 0

        self.agent_seed = agent_seed
        self.allocate_historical_metrics()

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

    def renew_power_conservative(self, date_in, cc_power_in_pib, deal_power_in_pib, duration_days, pledge_for_renewing, pledge_repayment_amount):
        """
        TODO: add to this ...
        However the most conservative picture is to assume constant onboarding and no effective renewal, in which case QAP 
        dips in the future as shown here.
        """
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

    def renew_power_optimistic(self, date_in, cc_power_in_pib, deal_power_in_pib, duration_days, pledge_for_renewing, pledge_repayment_amount):
        """
        TODO: add to this ...

         An optimistic framing is that although deals aren’t renewed in the same way sectors are extended, they may be re-onboarded 
         in the future at a similar rate, which could either be expressed as an effective renewal rate as it is currently, or alternatively 
         through growing onboarding. 
        """
        day_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]
        self.renewed_power[day_idx][0] += cc_power(cc_power_in_pib, duration_days)
        
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(date_in)].index[0]
        self.agent_info_df.loc[agent_df_idx, 'cc_renewed'] += cc_power_in_pib
        self.agent_info_df.loc[agent_df_idx, 'cc_renewed_duration'] = duration_days

        self.renewed_power[day_idx][1] += deal_power(deal_power_in_pib, duration_days)
        self.agent_info_df.loc[agent_df_idx, 'deal_renewed'] += deal_power_in_pib
        self.agent_info_df.loc[agent_df_idx, 'deal_renewed_duration'] = duration_days

        self._account_pledge_repayment_FIL(date_in, duration_days, pledge_for_renewing, pledge_repayment_amount)

    def renew_power(self, date_in, cc_power_in_pib, deal_power_in_pib, duration_days, pledge_for_renewing, pledge_repayment_amount):
        """
        In the Filecoin spec, only CC sectors are allowed to be renewed. However, in the simulation, we relax this constraint slightly and offer
        two different methods of computing renewals:
         1 - In the optimistic setting, renewals are computed for both QA and CC power. This is to capture the sentiment that as Deal sectors
            expire, even though an explicit renewal is not made, it is in effect with more deals coming online, or the deal being renewed
            through the normal channel of expire + re-onboard.
        2 - In the conservative setting, renewals are only computed for CC sectors.  NOTE that this is not properly implemented yet, b/c
            currently, the CC power is renewed.  However, the CC contains CC sectors & QA sectors (without the QA multiplier), so in effect
            this is not as conservative as you might expect if ONLY CC sectors were indeed being renewed.
        """
        if self.model.renewals_setting == 'optimistic':
            self.renew_power_optimistic(date_in, cc_power_in_pib, deal_power_in_pib, duration_days, pledge_for_renewing, pledge_repayment_amount)
        elif self.model.renewals_setting == 'conservative':
            self.renew_power_conservative(date_in, cc_power_in_pib, deal_power_in_pib, duration_days, pledge_for_renewing, pledge_repayment_amount)

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
            self.modeled_scheduled_expire_power[cc_expire_index][0] += today_onboarded_cc_power
        if deal_expire_index < self.sim_len_days:
            self.scheduled_expire_power[deal_expire_index][1] += today_onboarded_deal_power
            self.modeled_scheduled_expire_power[deal_expire_index][1] += today_onboarded_deal_power

        # do the same for renewals
        today_renewed_cc_power = self.renewed_power[self.current_day][0]
        today_renewed_deal_power = self.renewed_power[self.current_day][1]

        cc_expire_index = self.current_day + today_onboarded_cc_power.duration
        deal_expire_index = self.current_day + today_onboarded_deal_power.duration

        if cc_expire_index < self.sim_len_days:
            self.scheduled_expire_power[cc_expire_index][0] += today_renewed_cc_power
            self.modeled_scheduled_expire_power[cc_expire_index][0] += today_renewed_cc_power
        if deal_expire_index < self.sim_len_days:
            self.scheduled_expire_power[deal_expire_index][1] += today_renewed_deal_power
            self.modeled_scheduled_expire_power[deal_expire_index][1] += today_renewed_deal_power

        agent_df_idx = self.agent_info_df[self.agent_info_df['date'] == pd.to_datetime(self.current_date)].index[0]
        # account for today's onboarded power in cumulative stats
        self.agent_info_df.loc[agent_df_idx,'cum_cc_onboarded'] = self.agent_info_df.loc[agent_df_idx-1,'cum_cc_onboarded'] + today_onboarded_cc_power.pib
        self.agent_info_df.loc[agent_df_idx,'cum_deal_onboarded'] = self.agent_info_df.loc[agent_df_idx-1,'cum_deal_onboarded'] + today_onboarded_deal_power.pib
        
        # account for today's renewed power in cumulative stats
        self.agent_info_df.loc[agent_df_idx,'cum_cc_renewed'] = self.agent_info_df.loc[agent_df_idx-1,'cum_cc_renewed'] + today_renewed_cc_power.pib
        self.agent_info_df.loc[agent_df_idx,'cum_deal_renewed'] = self.agent_info_df.loc[agent_df_idx-1,'cum_deal_renewed'] + today_renewed_deal_power.pib

        # account for today's SE power in cumulative stats
        self.agent_info_df.loc[agent_df_idx, 'cum_cc_sched_expire'] = self.agent_info_df.loc[agent_df_idx-1, 'cum_cc_sched_expire'] + self.scheduled_expire_power[self.current_day][0].pib
        self.agent_info_df.loc[agent_df_idx, 'cum_deal_sched_expire'] = self.agent_info_df.loc[agent_df_idx-1, 'cum_deal_sched_expire'] + self.scheduled_expire_power[self.current_day][1].pib

        # account for today's terminations in cumulative stats
        self.agent_info_df.loc[agent_df_idx, 'cum_cc_terminated'] = self.agent_info_df.loc[agent_df_idx-1, 'cum_cc_terminated'] + self.terminated_power[self.current_day][0].pib
        self.agent_info_df.loc[agent_df_idx, 'cum_deal_terminated'] = self.agent_info_df.loc[agent_df_idx-1, 'cum_deal_terminated'] + self.terminated_power[self.current_day][1].pib

        self.current_day += 1
        self.current_date += timedelta(days=1)

    def disburse_rewards(self, date_in, reward):
        df_idx = self.accounting_df[self.accounting_df['date'] == date_in].index[0]
        self.accounting_df.loc[df_idx, 'reward_FIL'] += reward

    def get_active_qa_power_at_date(self, date_in):
        power_at_date = self.get_power_at_date(date_in)
        return (power_at_date['cum_deal_onboarded'] + power_at_date['cum_deal_renewed']) - (power_at_date['cum_deal_sched_expire'] + power_at_date['cum_deal_terminated'])

    def get_se_power_at_date(self, date_in):
        power_at_date = self.get_power_at_date(date_in)
        return {
            'se_cc_power': power_at_date['sched_expire_rb_pib'],
            'se_deal_power': power_at_date['sched_expire_qa_pib'],
        }

    def get_power_at_date(self, date_in):
        ii = (date_in - constants.NETWORK_DATA_START).days
        agent_df_idx = self.agent_info_df[self.agent_info_df['date'] == pd.to_datetime(date_in)].index[0]
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

            'cum_cc_onboarded': self.agent_info_df.loc[agent_df_idx, 'cum_cc_onboarded'],
            'cum_deal_onboarded': self.agent_info_df.loc[agent_df_idx, 'cum_deal_onboarded'],
            'cum_cc_renewed': self.agent_info_df.loc[agent_df_idx, 'cum_cc_renewed'],
            'cum_deal_renewed': self.agent_info_df.loc[agent_df_idx, 'cum_deal_renewed'],
            'cum_cc_sched_expire': self.agent_info_df.loc[agent_df_idx, 'cum_cc_sched_expire'],
            'cum_deal_sched_expire': self.agent_info_df.loc[agent_df_idx, 'cum_deal_sched_expire'],
            'cum_cc_terminated': self.agent_info_df.loc[agent_df_idx, 'cum_cc_terminated'],
            'cum_deal_terminated': self.agent_info_df.loc[agent_df_idx, 'cum_deal_terminated'],
            # 'scheduled_expire_pledge': self.scheduled_expire_pledge[ii],
        }
        return out_dict

    def allocate_historical_metrics(self):
        # need to distribute the power in the historical_power information from
        # day=0 to day=self.current_day-1
        historical_df = self.agent_seed['historical_power']
        future_se_df = self.agent_seed['future_se_power']

        # we can't vector assign b/c the power vectors are lists of objects, but 
        # this is premature optimization that we can revisit later
        global_ii = (historical_df.iloc[0]['date'] - constants.NETWORK_DATA_START).days
        cum_cc_onboarded, cum_cc_renewed, cum_cc_scheduled_expire, cum_cc_terminated = 0, 0, 0, 0
        cum_deal_onboarded, cum_deal_renewed, cum_deal_scheduled_expire, cum_deal_terminated = 0, 0, 0, 0
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(historical_df.iloc[0]['date'])].index[0]
        agent_df_idx_start = agent_df_idx
        for row_ii, row in historical_df.iterrows():
            day_onboarded_cc_power = row['day_onboarded_rb_power_pib']
            day_onboarded_deal_power = row['day_onboarded_qa_power_pib']
            extended_cc_power = row['extended_rb_pib']
            extended_deal_power = row['extended_qa_pib']
            scheduled_expire_cc_power = row['sched_expire_rb_pib']
            scheduled_expire_deal_power = row['sched_expire_qa_pib']
            terminated_cc_power = row['terminated_rb_pib']
            terminated_deal_power = row['terminated_qa_pib']

            self.onboarded_power[global_ii] = [
                cc_power(day_onboarded_cc_power), deal_power(day_onboarded_deal_power)
            ]
            self.renewed_power[global_ii] = [
                cc_power(extended_cc_power), deal_power(extended_deal_power)
            ]
            self.scheduled_expire_power[global_ii] = [
                cc_power(scheduled_expire_cc_power), deal_power(scheduled_expire_deal_power)
            ]
            self.terminated_power[global_ii] = [
                cc_power(terminated_cc_power), deal_power(terminated_deal_power)
            ]
            # self.scheduled_expire_pledge[global_ii] = row['total_pledge']

            if agent_df_idx == agent_df_idx_start:
                # 0.03008728516178749, -0.41849559534988146
                #cum_cc_onboarded = row['total_raw_power_eib'] * constants.EIB / constants.PIB # + historical_df.iloc[0:row_ii]['day_onboarded_rb_power_pib'].sum()
                #cum_deal_onboarded = row['total_qa_power_eib'] * constants.EIB / constants.PIB #+ historical_df.iloc[0:row_ii]['day_onboarded_qa_power_pib'].sum()

                cum_cc_renewed = historical_df.iloc[0:row_ii]['extended_rb_pib'].sum()
                cum_cc_scheduled_expire = historical_df.iloc[0:row_ii]['sched_expire_rb_pib'].sum()
                cum_cc_terminated = historical_df.iloc[0:row_ii]['terminated_rb_pib'].sum()
                
                cum_deal_renewed = historical_df.iloc[0:row_ii]['extended_qa_pib'].sum()
                cum_deal_scheduled_expire = historical_df.iloc[0:row_ii]['sched_expire_qa_pib'].sum()
                cum_deal_terminated =  historical_df.iloc[0:row_ii]['terminated_qa_pib'].sum()

                # 0.030087285161810584, -0.41849559534988146
                cum_cc_onboarded = row['total_raw_power_eib'] * constants.EIB / constants.PIB + cum_cc_renewed - cum_cc_scheduled_expire - cum_cc_terminated
                cum_deal_onboarded = row['total_qa_power_eib'] * constants.EIB / constants.PIB + cum_deal_renewed - cum_deal_scheduled_expire - cum_deal_terminated

                # print('seeding date:', historical_df.iloc[0]['date'], ' w/ total_cc_onboarded=', cum_cc_onboarded, ' total_qa_onboarded', cum_deal_onboarded)
                # raise Exception('stop')
            else:
                cum_cc_onboarded += day_onboarded_cc_power
                cum_cc_renewed += extended_cc_power
                cum_cc_scheduled_expire += scheduled_expire_cc_power
                cum_cc_terminated += terminated_cc_power

                cum_deal_onboarded += day_onboarded_deal_power
                cum_deal_renewed += extended_deal_power
                cum_deal_scheduled_expire += scheduled_expire_deal_power
                cum_deal_terminated += terminated_deal_power
            
            global_ii += 1
            # print("Seeding agent: %d from index=%d:%d" % (self.unique_id, ii_start, global_ii))
        
            self.agent_info_df.loc[agent_df_idx,'cum_cc_onboarded'] = cum_cc_onboarded
            self.agent_info_df.loc[agent_df_idx,'cum_cc_renewed'] = cum_cc_renewed
            self.agent_info_df.loc[agent_df_idx,'cum_cc_sched_expire'] = cum_cc_scheduled_expire
            self.agent_info_df.loc[agent_df_idx,'cum_cc_terminated'] = cum_cc_terminated

            self.agent_info_df.loc[agent_df_idx,'cum_deal_onboarded'] = cum_deal_onboarded
            self.agent_info_df.loc[agent_df_idx,'cum_deal_renewed'] = cum_deal_renewed
            self.agent_info_df.loc[agent_df_idx,'cum_deal_sched_expire'] = cum_deal_scheduled_expire
            self.agent_info_df.loc[agent_df_idx,'cum_deal_terminated'] = cum_deal_terminated
            agent_df_idx += 1

        # add in the SE power
        global_ii = (future_se_df.iloc[0]['date'] - constants.NETWORK_DATA_START).days
        for _, row in future_se_df.iterrows():
            self.scheduled_expire_power[global_ii] = [
                cc_power(row['sched_expire_rb_pib']),
                deal_power(row['sched_expire_qa_pib'])
            ]
            self.known_scheduled_expire_power[global_ii] = [
                cc_power(row['sched_expire_rb_pib']),
                deal_power(row['sched_expire_qa_pib'])
            ]

            global_ii += 1

        # add in the scheduled expirations
        # TODO: ensure dates are aligned?
        self.accounting_df['scheduled_pledge_release'] = self.agent_seed['scheduled_pledge_release'].values
    
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
        
        return np.sum(accounting_df_subset['reward_FIL']) - np.sum(accounting_df_subset['pledge_delta_FIL'] - np.sum(accounting_df_subset['termination_burned_FIL']))

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
    
    ## this function is for debugging only
    def _trace_modeled_power(self, onboarding_date, current_date):
        arr_idx = np.where(self.t == onboarding_date)[0][0]
        rb_onboarded_power = self.onboarded_power[arr_idx][0]
        qa_onboarded_power = self.onboarded_power[arr_idx][1]
        # get the rewards/sector on this date
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(onboarding_date)].index[0]
        rewards_per_sector_during_onboarding = self.model.filecoin_df.loc[filecoin_df_idx, 'day_rewards_per_sector']
        
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(onboarding_date)].index[0]
        associated_pledge_to_release = self.accounting_df.loc[accounting_df_idx, 'onboard_pledge_FIL']
        
        rb_active_power = rb_onboarded_power.pib
        qa_active_power = qa_onboarded_power.pib
        dur = rb_onboarded_power.duration
        iter_date = onboarding_date
        qa_rr_prod = 1  # for debugging only

        while iter_date < current_date:
            iter_date += timedelta(days=dur)
            if iter_date > current_date:
                break
            
            arr_idx += dur
            accounting_df_idx += dur

            rb_rr_at_dur = float(self.renewed_power[arr_idx][0].pib)/float(self.scheduled_expire_power[arr_idx][0].pib)
            qa_rr_at_dur = float(self.renewed_power[arr_idx][1].pib)/float(self.scheduled_expire_power[arr_idx][1].pib)
            assert rb_rr_at_dur <= 1.0
            assert qa_rr_at_dur <= 1.0
            rb_active_power *= rb_rr_at_dur
            qa_active_power *= qa_rr_at_dur

            # update the associated pledge to release based on what was renewed
            associated_pledge_to_release -= self.accounting_df.loc[accounting_df_idx, 'onboard_scheduled_pledge_release_FIL']
            associated_pledge_to_release += self.accounting_df.loc[accounting_df_idx, 'renew_pledge_FIL']
            qa_rr_prod *= qa_rr_at_dur

            dur = self.renewed_power[arr_idx][0].duration
            
        # a - convert active_power to sectors
        num_sectors_to_terminate = qa_active_power * constants.PIB / constants.SECTOR_SIZE
        # b - rewards/sector at time of onboarding * sectors * 90  --> 90 days is the termination fee
        termination_fee = int(rewards_per_sector_during_onboarding * num_sectors_to_terminate * 90)

        release_date = iter_date
        return rb_active_power, qa_active_power, termination_fee, associated_pledge_to_release, release_date
    
    def _exec_termination(self, onboarding_date, terminate_date, date_to_release, termination_fee, associated_pledge_to_release, rb_active_power, qa_active_power):
        accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(terminate_date)].index[0]
        self.accounting_df.loc[accounting_df_idx, 'termination_burned_FIL'] += termination_fee
        # update the termination statistics
        arr_idx = (terminate_date - constants.NETWORK_DATA_START).days
        self.terminated_power[arr_idx][0] += cc_power(rb_active_power)
        self.terminated_power[arr_idx][1] += deal_power(qa_active_power)

        # remove it from the SE power in the future when it was originally going to be expired
        date_to_release_idx = (date_to_release - constants.NETWORK_DATA_START).days
        self.scheduled_expire_power[date_to_release_idx][0] += cc_power(-rb_active_power)
        self.scheduled_expire_power[date_to_release_idx][1] += deal_power(-qa_active_power)
        
        # move when the pledge is released from the future to now
        self.accounting_df.loc[accounting_df_idx, 'scheduled_pledge_release'] += associated_pledge_to_release
        date_to = terminate_date
        if date_to_release is not None:
            date_to_release_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(date_to_release)].index
            if len(date_to_release_idx) > 0:
                self.accounting_df.loc[date_to_release_idx[0], 'scheduled_pledge_release'] -= associated_pledge_to_release
        
        self.model._release_pledge(date_to_release, date_to, associated_pledge_to_release)

        # take care of reward vesting
        num_days_since_onboard = (terminate_date - onboarding_date).days
        if num_days_since_onboard < 180:
            # we have some unvested rewards that need to be removed from the agent's rewards
            accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(onboarding_date)].index[0]
            full_rewards_for_power = self.accounting_df.loc[accounting_df_idx, 'full_reward_for_power_FIL']
            num_days_remaining_vest = 180-num_days_since_onboard
            vest_rewards_per_day = full_rewards_for_power*0.75/180

            accounting_df_idx = self.accounting_df[pd.to_datetime(self.accounting_df['date']) == pd.to_datetime(terminate_date)].index[0]
            self.accounting_df.loc[accounting_df_idx:accounting_df_idx+num_days_remaining_vest, 'reward_FIL'] -= vest_rewards_per_day

    def terminate_modeled_power(self, onboarding_date, terminate_date=None):
        # terminates power which was onboarded on `onboarding_date` at terminate_date
        if terminate_date is None:
            terminate_date = self.current_date

        rb_active_power, qa_active_power, termination_fee, associated_pledge_to_release, date_to_release = self._trace_modeled_power(onboarding_date, terminate_date)
        # print(onboarding_date, terminate_date, date_to_release, rb_active_power, qa_active_power, termination_fee, associated_pledge_to_release, final_duration)
        self._exec_termination(onboarding_date, terminate_date, date_to_release, termination_fee, associated_pledge_to_release, rb_active_power, qa_active_power)
    
    def terminate_all_modeled_power(self, terminate_date=None):
        current_date = self.start_date
        while current_date < terminate_date:
            self.terminate_modeled_power(current_date, terminate_date)
            current_date += timedelta(days=1)

    def terminate_all_known_power(self, terminate_date=None):
        # get the known power which is still active at terminate_date
        if terminate_date is None:
            terminate_date = self.current_date
        assert terminate_date >= self.start_date
        terminate_date_m1 = terminate_date - timedelta(days=1)

        # the algorithm works as follows.  Since we don't know the exact time of onboarding the known power,
        # we compute the active modeled power and pledge to be released due to modeled power onboarding.
        # we subtract that from the cumulative, which should give us the active known power and pledge to be released
        current_date = self.start_date
        total_rb_modeled_power_active = 0
        total_qa_modeled_power_active = 0
        total_modeled_pledge_to_release = 0  # I *think* this represents the total locked due to modeled power!
        while current_date < terminate_date:
            rb_active_power, qa_active_power, termination_fee, associated_pledge_to_release, release_date = self._trace_modeled_power(current_date, terminate_date)

            total_rb_modeled_power_active += rb_active_power
            total_qa_modeled_power_active += qa_active_power
            total_modeled_pledge_to_release += associated_pledge_to_release

            current_date += timedelta(days=1)

        # this is the sum of the known onboarded power until terminate_date - SE expire power at terminate_date
        agent_info_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(terminate_date_m1)].index[0]
        agent_at_terminate_date_stats = self.agent_info_df.loc[agent_info_df_idx]
        
        total_current_rb_active_power = agent_at_terminate_date_stats['cum_cc_onboarded'] + agent_at_terminate_date_stats['cum_cc_renewed'] - agent_at_terminate_date_stats['cum_cc_sched_expire'] - agent_at_terminate_date_stats['cum_cc_terminated']
        total_current_qa_active_power = agent_at_terminate_date_stats['cum_deal_onboarded'] + agent_at_terminate_date_stats['cum_deal_renewed'] - agent_at_terminate_date_stats['cum_deal_sched_expire'] - agent_at_terminate_date_stats['cum_deal_terminated']
        
        known_rb_active_power = total_current_rb_active_power - total_rb_modeled_power_active
        known_qa_active_power = total_current_qa_active_power - total_qa_modeled_power_active
        # the pledge to be released at the terminate date due to known power can be computed as:
        #  total_locked@terminate - locked@terminate[due to modeled power]
        _, agentid2powerproportion = self.model._get_agent_power_proportion(update_day=self.current_day-1)  # havent aggregated today's decisions yet
        # total_locked_at_terminate_date = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(terminate_date_m1)]['network_locked'].values[0] * self.agent_seed['agent_power_pct']
        print(agentid2powerproportion)
        total_locked_at_terminate_date = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(terminate_date_m1)]['network_locked'].values[0] * agentid2powerproportion[self.unique_id]
        known_pledge_to_release = total_locked_at_terminate_date - total_modeled_pledge_to_release

        # print(self.agent_info_df.loc[agent_info_df_idx-1])
        # print(agent_at_terminate_date_stats)
        print(total_current_rb_active_power, total_current_qa_active_power)
        print(total_rb_modeled_power_active, total_qa_modeled_power_active)
        print(known_rb_active_power, known_qa_active_power)
        print(total_locked_at_terminate_date, total_modeled_pledge_to_release, known_pledge_to_release)

        # release the pledge proportional to the SE
        known_scheduled_pledge_release_df = self.agent_seed['scheduled_pledge_release']
        known_scheduled_pledge_release_after_terminate_df = known_scheduled_pledge_release_df[known_scheduled_pledge_release_df['date'] > terminate_date_m1]
        ix = np.where(known_scheduled_pledge_release_after_terminate_df['scheduled_pledge_release'] == 0)[0][0]
        end_of_known_power_date = known_scheduled_pledge_release_after_terminate_df.iloc[ix]['date']
        known_scheduled_pledge_release_filtered = known_scheduled_pledge_release_after_terminate_df[known_scheduled_pledge_release_after_terminate_df['date'] < end_of_known_power_date]
        current_date = terminate_date

        # NOTE: release_pct_vector has a big effect on how the locked trajectory looks.
        release_pct_vector = (known_scheduled_pledge_release_filtered['scheduled_pledge_release'] / known_scheduled_pledge_release_filtered['scheduled_pledge_release'].sum()).values
        
        known_power_onboarding_date_approx = constants.NETWORK_DATA_START
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(known_power_onboarding_date_approx)].index[0]
        rewards_per_sector_during_onboarding = self.model.filecoin_df.loc[filecoin_df_idx, 'day_rewards_per_sector']
        ix = 0
        while current_date < end_of_known_power_date:
            release_pct = release_pct_vector[ix]
            ix += 1

            rb_power_to_release = float(known_rb_active_power) * release_pct
            qa_power_to_release = float(known_qa_active_power) * release_pct
            pledge_to_release = int(float(known_pledge_to_release) * release_pct)

            num_sectors_to_terminate = known_qa_active_power * constants.PIB / constants.SECTOR_SIZE
            # b - rewards/sector at time of onboarding * sectors * 90  --> 90 days is the termination fee
            termination_fee = int(rewards_per_sector_during_onboarding * num_sectors_to_terminate * 90)

            self._exec_termination(known_power_onboarding_date_approx, terminate_date, current_date, termination_fee, pledge_to_release, rb_power_to_release, qa_power_to_release)
            current_date += timedelta(days=1)
        
    def save_data(self, output_dir):
        accounting_fp = os.path.join(output_dir, 'agent_%d_accounting_info.csv' % (self.unique_id,))
        self.accounting_df.to_csv(accounting_fp, index=False)

        agent_info_fp = os.path.join(output_dir, 'agent_%d_agent_info.csv' % (self.unique_id,))
        self.agent_info_df.to_csv(agent_info_fp, index=False)