import mesa
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from datetime import datetime, timedelta

from mechafil import data, vesting, minting

from . import constants
from . import sp_agent


def solve_geometric(a, n):
    # see: https://math.stackexchange.com/a/2174287
    def f(r, a, n):
        # the geometric series 
        return a*(np.power(r,n)-1)/(r-1) - 1
    init_guess = 0.5
    soln = fsolve(f, init_guess, args=(a, n))
    r = soln[0]
    return r

class FilecoinModel(mesa.Model):
    def __init__(self, n, start_date, end_date):
        self.num_agents = n
        self.schedule = mesa.time.SimultaneousActivation(self)

        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date
        self.current_day = (self.current_date - constants.NETWORK_DATA_START).days

        self.agents = []
        self.rbp0 = None
        self.qap0 = None
        self.seed_agents()

        self.initialize_network_description_df()
        self.fast_forward_to_simulation_start()

    def download_historical_data(self):
        historical_stats = data.get_historical_network_stats(
            constants.NETWORK_DATA_START,
            self.start_date,
            self.end_date
        )
        scheduled_df = data.query_sector_expirations(constants.NETWORK_DATA_START, self.end_date)
        # get the fields necessary to seed the agents into a separate dataframe that is time-aligned
        historical_stats['date'] = pd.to_datetime(historical_stats['date']).dt.date
        scheduled_df['date'] = scheduled_df['date'].dt.date
        merged_df = historical_stats.merge(scheduled_df, on='date', how='inner')
        
        # NOTE: consider using scheduled_expire_rb rather than total_rb??
        df_historical = merged_df[
            [
                'date', 
                'day_onboarded_rb_power_pib', 'extended_rb', 'total_rb', 'terminated_rb',
                'day_onboarded_qa_power_pib', 'extended_qa', 'total_qa', 'terminated_qa',
            ]
        ]
        final_date_historical = historical_stats.iloc[-1]['date']
        df_future = scheduled_df[scheduled_df['date'] >= final_date_historical][['date', 'total_rb', 'total_qa']]

        self.rbp0 = merged_df.iloc[0]['total_raw_power_eib']
        self.qap0 = merged_df.iloc[0]['total_qa_power_eib']
        self.max_date_se_power = df_future.iloc[-1]['date']

        return df_historical, df_future

    def seed_agents(self):
        df_historical, df_future = self.download_historical_data()
        
        # use a geometric-series to determine the proportion of power that goes
        # to each agent
        a = 0.2
        r = solve_geometric(a, self.num_agents)
        
        for ii in range(self.num_agents):
            agent_power_pct = a*(r**ii)
            agent_historical_df = df_historical.drop('date', axis=1) * agent_power_pct
            agent_historical_df['date'] = df_historical['date']
            agent_future_df = df_future.drop('date', axis=1) * agent_power_pct
            agent_future_df['date'] = df_future['date']
            agent_seed = {
                'historical_power': agent_historical_df,
                'future_se_power': agent_future_df
            }
            agent = sp_agent.SPAgent(ii, agent_seed, 
                                     self.start_date, self.end_date)

            self.schedule.add(agent)
            self.agents.append(
                {
                    'agent_power_pct': agent_power_pct,
                    'agent': agent
                }
            )

    def initialize_network_description_df(self):
        self.filecoin_df = pd.DataFrame()
        
        # precompute columns which do not depend on inputs
        self.filecoin_df['date'] = pd.date_range(constants.NETWORK_DATA_START, self.end_date, freq='D')[:-1]
        self.filecoin_df['date'] = self.filecoin_df['date'].dt.date
        self.filecoin_df['days'] = np.arange(1, len(self.filecoin_df) + 1)  # should this start w/ 0 or 1??

        self.filecoin_df['network_baseline'] = minting.compute_baseline_power_array(constants.NETWORK_DATA_START, self.end_date)
        vest_df = vesting.compute_vesting_trajectory_df(constants.NETWORK_DATA_START, self.end_date)
        
        self.filecoin_df = self.filecoin_df.merge(vest_df, on='date', how='inner')
        

    def compute_macro(self, date_in):
        # compute Macro econometrics (Power statistics)
        total_rb_delta, total_qa_delta = 0, 0
        total_onboarded_rb_delta, total_renewed_rb_delta, total_se_rb_delta, total_terminated_rb_delta = 0, 0, 0, 0
        total_onboarded_qa_delta, total_renewed_qa_delta, total_se_qa_delta, total_terminated_qa_delta = 0, 0, 0, 0
        for agent_info in self.agents:
            agent = agent_info['agent']
            agent_day_power_stats = agent.get_power_at_date(date_in)
            
            total_onboarded_rb_delta += agent_day_power_stats['day_onboarded_rb_power_pib']
            total_onboarded_qa_delta += agent_day_power_stats['day_onboarded_qa_power_pib']
            
            total_renewed_rb_delta += agent_day_power_stats['extended_rb']
            total_renewed_qa_delta += agent_day_power_stats['extended_qa']

            total_se_rb_delta += agent_day_power_stats['total_rb']
            total_se_qa_delta += agent_day_power_stats['total_qa']

            total_terminated_rb_delta += agent_day_power_stats['terminated_rb']
            total_terminated_qa_delta += agent_day_power_stats['terminated_qa']

        total_rb_delta += (total_onboarded_rb_delta + total_renewed_rb_delta - total_se_rb_delta - total_terminated_rb_delta)
        total_qa_delta += (total_onboarded_qa_delta + total_renewed_qa_delta - total_se_qa_delta - total_terminated_qa_delta)

        out_dict = {
            'date': date_in,
            'day_onboarded_rbp_pib': total_onboarded_rb_delta,
            'day_onboarded_qap_pib': total_onboarded_qa_delta,
            'day_renewed_rbp_pib': total_renewed_rb_delta,
            'day_renewed_qap_pib': total_renewed_qa_delta,
            'day_sched_expire_rbp_pib': total_se_rb_delta,
            'day_sched_expire_qap_pib': total_se_qa_delta,
            'day_terminated_rbp_pib': total_terminated_rb_delta,
            'day_terminated_qap_pib': total_terminated_qa_delta,
            'day_network_rbp_pib': total_rb_delta,
            'day_network_qap_pib': total_qa_delta
        }
        return out_dict

    def fast_forward_to_simulation_start(self):
        current_date = constants.NETWORK_DATA_START
        day_power_stats_vec = []
        while current_date < self.start_date:
            day_power_stats = self.compute_macro(current_date)
            day_power_stats_vec.append(day_power_stats)

            current_date += timedelta(days=1)
        power_stats_df = pd.DataFrame(day_power_stats_vec)
        cur_idx = power_stats_df.index[-1]

        # create cumulative statistics which are needed to compute minting
        power_stats_df['total_raw_power_eib'] = power_stats_df['day_network_rbp_pib'].cumsum() / 1024.0 + self.rbp0
        power_stats_df['total_qa_power_eib'] = power_stats_df['day_network_qap_pib'].cumsum() / 1024.0  + self.qap0

        ##########################################################################################
        # TODO: encapsulate this into a function b/c it needs to be done iteratively in the model step function
        filecoin_df_subset = self.filecoin_df[self.filecoin_df['date'] < self.start_date]
        power_stats_df["cum_simple_reward"] = self.filecoin_df["days"].pipe(minting.cum_simple_minting)
        power_stats_df['capped_power'] = (constants.EIB*power_stats_df['total_raw_power_eib']).clip(upper=filecoin_df_subset['network_baseline'])
        power_stats_df['cum_capped_power'] = power_stats_df['capped_power'].cumsum()
        power_stats_df['network_time'] = power_stats_df['cum_capped_power'].pipe(minting.network_time)
        power_stats_df['cum_baseline_reward'] = power_stats_df['network_time'].pipe(minting.cum_baseline_reward)
        power_stats_df['cum_network_reward'] = power_stats_df['cum_baseline_reward'] + power_stats_df['cum_simple_reward']
        power_stats_df['day_network_reward'] = power_stats_df['cum_network_reward'].diff().fillna(method='backfill')
        ##########################################################################################

        # concatenate w/ NA for rest of the simulation so that the merge doesn't delete the data in the master DF
        remaining_sim_len = (self.end_date - self.start_date).days
        remaining_power_stats_df = pd.DataFrame(np.nan, index=range(remaining_sim_len), columns=power_stats_df.columns)
        remaining_power_stats_df['date'] = pd.date_range(self.start_date, self.end_date, freq='D')[:-1]

        power_stats_df = pd.concat([power_stats_df, remaining_power_stats_df], ignore_index=True)

        # for proper merging, we need to convert to datetime
        power_stats_df['date'] = pd.to_datetime(power_stats_df['date'])
        self.filecoin_df['date'] = pd.to_datetime(self.filecoin_df['date'])

        # merge this into the master filecoin description dataframe
        self.filecoin_df = self.filecoin_df.merge(power_stats_df, on='date', how='outer')
        self.filecoin_df['date'] = self.filecoin_df['date'].dt.date

        # add in future SE power
        se_power_stats_vec = []
        while current_date < self.max_date_se_power:
            se_power_stats = self.compute_macro(current_date)
            se_power_stats_vec.append(se_power_stats)

            current_date += timedelta(days=1)
        se_power_stats_df = pd.DataFrame(se_power_stats_vec)

        l = len(se_power_stats_df)
        self.filecoin_df.loc[cur_idx+1:cur_idx+l, ['day_sched_expire_rbp_pib']] = se_power_stats_df['day_sched_expire_rbp_pib'].values
        self.filecoin_df.loc[cur_idx+1:cur_idx+l, ['day_sched_expire_qap_pib']] = se_power_stats_df['day_sched_expire_qap_pib'].values


    def step(self):
        # step agents
        self.schedule.step()

        day_macro_info = self.compute_macro(self.current_date)

        # update cumulative macro metrics

        # compute minting

        # compute circulating supply

        # record macro-updates that will be used for analysis

        # increment counters
        self.current_date += timedelta(days=1)