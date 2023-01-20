import mesa
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from datetime import timedelta

from mechafil import data, vesting, minting, supply

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
    def __init__(self, n, start_date, end_date, agent_types=None):
        self.num_agents = n
        self.schedule = mesa.time.SimultaneousActivation(self)

        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date

        self.start_day = (self.start_date - constants.NETWORK_DATA_START).days
        self.current_day = (self.current_date - constants.NETWORK_DATA_START).days

        self.agents = []
        self.rbp0 = None
        self.qap0 = None
        self.seed_agents(agent_types=agent_types)

        self.initialize_network_description_df()
        self.fast_forward_to_simulation_start()
        print('Current Date=', self.current_date)

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
        # df_future = scheduled_df[scheduled_df['date'] >= final_date_historical][['date', 'total_rb', 'total_qa', 'total_pledge']]
        df_future = scheduled_df[scheduled_df['date'] >= final_date_historical][['date', 'total_rb', 'total_qa']]

        self.rbp0 = merged_df.iloc[0]['total_raw_power_eib']
        self.qap0 = merged_df.iloc[0]['total_qa_power_eib']
        self.max_date_se_power = df_future.iloc[-1]['date']
        ## TODO: we need to change this out so that pledge is tracked by individual agent
        #  but in the interest of getting something quickly running, we will use a static
        #  duration when computing the circ-supply. This will be inconsistent w/ the power
        #  module so it will be a good idea to change this soon.
        # this vector starts from the first day of the simulation
        self.known_scheduled_pledge_release_full_vec = scheduled_df[scheduled_df['date'] >= final_date_historical]["total_pledge"].values
        # print(self.known_scheduled_pledge_release_full_vec[0:5], self.known_scheduled_pledge_release_full_vec[-5:])
        self.duration = 365
        self.lock_target = 0.3

        self.zero_cum_capped_power = data.get_cum_capped_rb_power(constants.NETWORK_DATA_START)

        return df_historical, df_future

    def seed_agents(self, agent_types=None):
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
            if agent_types is not None:
                agent_cls = agent_types[ii]
            else:
                agent_cls = sp_agent.SPAgent
            # TODO: need a better way to instantiate agents of interest
            # likely, we need to have different seed functions as a helper
            # utility
            agent = agent_cls(ii, agent_seed, 
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
        days_offset = (constants.NETWORK_DATA_START - constants.NETWORK_START).days
        self.filecoin_df['days'] = np.arange(days_offset, len(self.filecoin_df)+days_offset)

        self.filecoin_df['network_baseline'] = minting.compute_baseline_power_array(constants.NETWORK_DATA_START, self.end_date)
        vest_df = vesting.compute_vesting_trajectory_df(constants.NETWORK_DATA_START, self.end_date)
        self.filecoin_df["cum_simple_reward"] = self.filecoin_df["days"].pipe(minting.cum_simple_minting)
        
        self.filecoin_df = self.filecoin_df.merge(vest_df, on='date', how='inner')
        

    def compute_macro(self, date_in):
        # compute Macro econometrics (Power statistics)
        total_rb_delta, total_qa_delta = 0, 0
        total_onboarded_rb_delta, total_renewed_rb_delta, total_se_rb_delta, total_terminated_rb_delta = 0, 0, 0, 0
        total_onboarded_qa_delta, total_renewed_qa_delta, total_se_qa_delta, total_terminated_qa_delta = 0, 0, 0, 0
        # total_scheduled_expire_pledge = 0
        # print('Computing Macro for date:', date_in)
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

            # total_scheduled_expire_pledge += agent_day_power_stats['scheduled_expire_pledge']

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
            'day_network_qap_pib': total_qa_delta,
            # 'day_scheduled_expire_pledge': total_scheduled_expire_pledge,
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
        final_historical_data_idx = power_stats_df.index[-1]

        # create cumulative statistics which are needed to compute minting
        power_stats_df['total_raw_power_eib'] = power_stats_df['day_network_rbp_pib'].cumsum() / 1024.0 + self.rbp0
        power_stats_df['total_qa_power_eib'] = power_stats_df['day_network_qap_pib'].cumsum() / 1024.0  + self.qap0

        # ##########################################################################################
        # NOTE: encapsulate this into a function b/c it needs to be done iteratively in the model step function
        filecoin_df_subset = self.filecoin_df[self.filecoin_df['date'] < self.start_date]
        # TODO: better to check the actual dates rather than lengths
        assert len(filecoin_df_subset) == len(power_stats_df)
        assert power_stats_df.iloc[0]['date'] == filecoin_df_subset.iloc[0]['date']
        assert power_stats_df.iloc[-1]['date'] == filecoin_df_subset.iloc[-1]['date']
        
        power_stats_df['capped_power'] = (constants.EIB*power_stats_df['total_raw_power_eib']).clip(upper=filecoin_df_subset['network_baseline'])
        power_stats_df['cum_capped_power'] = power_stats_df['capped_power'].cumsum() + self.zero_cum_capped_power
        power_stats_df['network_time'] = power_stats_df['cum_capped_power'].pipe(minting.network_time)
        power_stats_df['cum_baseline_reward'] = power_stats_df['network_time'].pipe(minting.cum_baseline_reward)
        power_stats_df['cum_network_reward'] = power_stats_df['cum_baseline_reward'].values + filecoin_df_subset['cum_simple_reward'].values
        power_stats_df['day_network_reward'] = power_stats_df['cum_network_reward'].diff().fillna(method='backfill')
        # ##########################################################################################

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
        self.filecoin_df.loc[final_historical_data_idx+1:final_historical_data_idx+l, ['day_sched_expire_rbp_pib']] = se_power_stats_df['day_sched_expire_rbp_pib'].values
        self.filecoin_df.loc[final_historical_data_idx+1:final_historical_data_idx+l, ['day_sched_expire_qap_pib']] = se_power_stats_df['day_sched_expire_qap_pib'].values

        # initialize the circulating supply
        supply_df = data.query_supply_stats(constants.NETWORK_DATA_START, self.start_date)
        start_idx = self.filecoin_df[self.filecoin_df['date'] == supply_df.iloc[0]['date']].index[0]
        end_idx = self.filecoin_df[self.filecoin_df['date'] == supply_df.iloc[-1]['date']].index[0]
        # self.filecoin_df.loc[start_idx:end_idx, ['circ_supply']] = supply_df['circulating_fil'].values
        # self.filecoin_df.loc[start_idx:end_idx, ['cum_network_reward']] = supply_df['mined_fil'].values  # this was computed above
        self.filecoin_df.loc[start_idx:end_idx, 'total_vest'] = supply_df['vested_fil'].values
        self.filecoin_df.loc[start_idx:end_idx, 'network_locked'] = supply_df['locked_fil'].values
        self.filecoin_df.loc[start_idx:end_idx, 'network_gas_burn'] = supply_df['burnt_fil'].values
        self.filecoin_df['disbursed_reserve'] = (17066618961773411890063046 * 10**-18)  # constant across time.
        
        # recompute it to be consistent ... but need to figure out why there is a discrepancy between minting
        self.filecoin_df.loc[start_idx:end_idx, 'circ_supply'] = (
            self.filecoin_df.loc[start_idx:end_idx, 'disbursed_reserve']
            + self.filecoin_df.loc[start_idx:end_idx, 'cum_network_reward']  # from the minting_model
            + self.filecoin_df.loc[start_idx:end_idx, 'total_vest']  # from vesting_model
            - self.filecoin_df.loc[start_idx:end_idx, 'network_locked']  # from simulation loop
            - self.filecoin_df.loc[start_idx:end_idx, 'network_gas_burn']  # comes from user inputs
        )
        
        # internal CS metrics are initialized to 0 and only filled after the simulation start date
        self.filecoin_df['day_locked_pledge'] = 0
        self.filecoin_df['day_renewed_pledge'] = 0
        self.filecoin_df['network_locked_pledge'] = 0
        self.filecoin_df['network_locked_reward'] = 0
        self.daily_burnt_fil = supply_df["burnt_fil"].diff().mean()
        print('Initialized CircSupply from: ', start_idx, end_idx)
        
        locked_fil_zero = self.filecoin_df.loc[final_historical_data_idx, ["network_locked"]].values[0]
        self.filecoin_df.loc[final_historical_data_idx+1, ["network_locked_pledge"]] = locked_fil_zero / 2.0
        self.filecoin_df.loc[final_historical_data_idx+1, ["network_locked_reward"]] = locked_fil_zero / 2.0
        self.filecoin_df.loc[final_historical_data_idx+1, ["network_locked"]] = locked_fil_zero

    def _update_power_metrics(self, day_macro_info, day_idx=None):
        day_idx = self.current_day if day_idx is None else day_idx

        self.filecoin_df.loc[day_idx, 'day_onboarded_rbp_pib'] = day_macro_info['day_onboarded_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_onboarded_qap_pib'] = day_macro_info['day_onboarded_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_renewed_rbp_pib'] = day_macro_info['day_renewed_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_renewed_qap_pib'] = day_macro_info['day_renewed_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_sched_expire_rbp_pib'] = day_macro_info['day_sched_expire_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_sched_expire_qap_pib'] = day_macro_info['day_sched_expire_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_terminated_rbp_pib'] = day_macro_info['day_terminated_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_terminated_qap_pib'] = day_macro_info['day_terminated_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_network_rbp_pib'] = day_macro_info['day_network_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_network_qap_pib'] = day_macro_info['day_network_qap_pib']

        self.filecoin_df.loc[day_idx, 'total_raw_power_eib'] = self.filecoin_df.loc[day_idx, 'day_network_rbp_pib'] / 1024.0 + self.filecoin_df.loc[day_idx-1, 'total_raw_power_eib']
        self.filecoin_df.loc[day_idx, 'total_qa_power_eib'] = self.filecoin_df.loc[day_idx, 'day_network_qap_pib'] / 1024.0 + self.filecoin_df.loc[day_idx-1, 'total_qa_power_eib']

    def _update_minting(self, day_idx=None):
        day_idx = self.current_day if day_idx is None else day_idx
        baseline_pwr = self.filecoin_df.loc[day_idx, 'network_baseline']

        capped_power = min(constants.EIB*self.filecoin_df.loc[day_idx, 'total_raw_power_eib'], baseline_pwr)
        cum_capped_power = capped_power + self.filecoin_df.loc[day_idx-1, 'cum_capped_power']
        self.filecoin_df.loc[day_idx, 'capped_power'] = capped_power
        self.filecoin_df.loc[day_idx, 'cum_capped_power'] = cum_capped_power
        network_time = minting.network_time(cum_capped_power)
        self.filecoin_df.loc[day_idx, 'network_time'] = network_time
        cum_baseline_reward = minting.cum_baseline_reward(network_time)
        self.filecoin_df.loc[day_idx, 'cum_baseline_reward'] = cum_baseline_reward
        cum_network_reward = cum_baseline_reward + self.filecoin_df.loc[day_idx, 'cum_simple_reward']
        self.filecoin_df.loc[day_idx, 'cum_network_reward'] = cum_network_reward
        self.filecoin_df.loc[day_idx, 'day_network_reward'] = cum_network_reward - self.filecoin_df.loc[day_idx-1, 'cum_network_reward']

    def _update_circulating_supply(self, update_day=None):
        day_idx = self.current_day if update_day is None else update_day

        day_pledge_locked_vec = self.filecoin_df["day_locked_pledge"].values
        day_onboarded_power_QAP = self.filecoin_df.iloc[day_idx]["day_onboarded_qap_pib"] * constants.PIB   # in bytes
        day_renewed_power_QAP = self.filecoin_df.iloc[day_idx]["day_renewed_qap_pib"] * constants.PIB       # in bytes
        network_QAP = self.filecoin_df.iloc[day_idx]["total_qa_power_eib"] * constants.EIB                  # in bytes
        network_baseline = self.filecoin_df.iloc[day_idx]["network_baseline"]                               # in bytes
        day_network_reward = self.filecoin_df.iloc[day_idx]["day_network_reward"]
        renewal_rate = self.filecoin_df.iloc[day_idx]["day_renewed_rbp_pib"] / self.filecoin_df.iloc[day_idx]["day_sched_expire_rbp_pib"]

        prev_network_locked_reward = self.filecoin_df.iloc[day_idx-1]["network_locked_reward"]
        prev_network_locked_pledge = self.filecoin_df.iloc[day_idx-1]["network_locked_pledge"]
        prev_network_locked = self.filecoin_df.iloc[day_idx-1]["network_locked"]

        prev_circ_supply = self.filecoin_df["circ_supply"].iloc[day_idx-1]

        scheduled_pledge_release = supply.get_day_schedule_pledge_release(
            day_idx-self.start_day,  # this is a hack to get the correct index, but very confusing. need to redo this!
            0,
            day_pledge_locked_vec,
            self.known_scheduled_pledge_release_full_vec,
            self.duration,
        )
        pledge_delta = supply.compute_day_delta_pledge(
            day_network_reward,
            prev_circ_supply,
            day_onboarded_power_QAP,
            day_renewed_power_QAP,
            network_QAP,
            network_baseline,
            renewal_rate,
            scheduled_pledge_release,
            self.lock_target,
        )
        # Get total locked pledge (needed for future day_locked_pledge)
        day_locked_pledge, day_renewed_pledge = supply.compute_day_locked_pledge(
            day_network_reward,
            prev_circ_supply,
            day_onboarded_power_QAP,
            day_renewed_power_QAP,
            network_QAP,
            network_baseline,
            renewal_rate,
            scheduled_pledge_release,
            self.lock_target,
        )
        # Compute daily change in block rewards collateral
        day_locked_rewards = supply.compute_day_locked_rewards(day_network_reward)
        day_reward_release = supply.compute_day_reward_release(prev_network_locked_reward)
        reward_delta = day_locked_rewards - day_reward_release
        
        # Update dataframe
        self.filecoin_df.loc[day_idx, "day_locked_pledge"] = day_locked_pledge
        self.filecoin_df.loc[day_idx, "day_renewed_pledge"] = day_renewed_pledge
        self.filecoin_df.loc[day_idx, "network_locked_pledge"] = (
            prev_network_locked_pledge + pledge_delta
        )
        self.filecoin_df.loc[day_idx, "network_locked_reward"] = (
            prev_network_locked_reward + reward_delta
        )
        self.filecoin_df.loc[day_idx, "network_locked"] = (
            prev_network_locked + pledge_delta + reward_delta
        )
        # Update gas burnt
        # if self.filecoin_df.loc[day_idx, "network_gas_burn"] == 0.0:
        self.filecoin_df["network_gas_burn"].iloc[day_idx] = (
            self.filecoin_df["network_gas_burn"].iloc[day_idx - 1] + self.daily_burnt_fil
        )
        # Find circulating supply balance and update
        circ_supply = (
            self.filecoin_df["disbursed_reserve"].iloc[day_idx]  # from initialise_circulating_supply_df
            + self.filecoin_df["cum_network_reward"].iloc[day_idx]  # from the minting_model
            + self.filecoin_df["total_vest"].iloc[day_idx]  # from vesting_model
            - self.filecoin_df["network_locked"].iloc[day_idx]  # from simulation loop
            - self.filecoin_df["network_gas_burn"].iloc[day_idx]  # comes from user inputs
        )
        self.filecoin_df.loc[day_idx, "circ_supply"] = max(circ_supply, 0)

    def step(self):
        # step agents
        self.schedule.step()

        day_macro_info = self.compute_macro(self.current_date)
        self._update_power_metrics(day_macro_info)
        self._update_minting()
        self._update_circulating_supply()
        # update any other inputs to agents

        # increment counters
        self.current_date += timedelta(days=1)
        self.current_day += 1