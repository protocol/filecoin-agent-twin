import mesa
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from datetime import timedelta
import math

from mechafil import data, vesting, minting, supply, locking

from . import constants
from . import sp_agent
from . import rewards_per_sector_process
from . import price_process
from . import capital_inflow_process


def solve_geometric(a, n):
    # see: https://math.stackexchange.com/a/2174287
    def f(r, a, n):
        # the geometric series 
        return a*(np.power(r,n)-1)/(r-1) - 1
    init_guess = 0.5
    soln = fsolve(f, init_guess, args=(a, n))
    r = soln[0]
    return r

def distribute_agent_power_geometric_series(num_agents, a=0.2):
    # use a geometric-series to determine the proportion of power that goes
    # to each agent
    if num_agents == 1:
        return [1.0]
    
    r = solve_geometric(a, num_agents)

    agent_power_distributions = []
    for i in range(num_agents):
        agent_power_pct = a*(r**i)
        agent_power_distributions.append(agent_power_pct)
    return agent_power_distributions


def apply_qa_multiplier(power_in, 
                        fil_plus_multipler=constants.FIL_PLUS_MULTIPLER, 
                        date_in=None, sdm=None, sdm_kwargs=None):
    if sdm is None:
        return power_in * fil_plus_multipler
    else:
        sdm_multiplier = sdm(power_in, **sdm_kwargs)
        return power_in * sdm_multiplier * fil_plus_multipler


class FilecoinModel(mesa.Model):
    def __init__(self, n, start_date, end_date, 
                 agent_types=None, agent_kwargs_list=None, agent_power_distributions=None,
                 compute_cs_from_networkdatastart=True, use_historical_gas=False,
                 price_process_kwargs=None, minting_process_kwargs=None, capital_inflow_process_kwargs=None,
                 capital_inflow_distribution_policy=None, capital_inflow_distribution_policy_kwargs=None,
                 random_seed=1234):
        """
        start_date: the start date of the simulation
        end_date: the end date of the simulation
        agent_types: a vector of the types of agents to instantiate, if None then the 
                     default is to instantiate all agents as SPAgent
        agent_kwargs_list: a list of dictionaries, each dictionary contains keywords to configure
                            the instantiated agent.
        agent_power_distributions: a vector of the proportion of power that goes to each agent. If None,
                            will be computed using the default function `distribute_agent_power_geometric_series`
        compute_cs_from_networkdatastart: if True, then the circulating supply is computed from the
                               start of network data (2021-03-15). If False, then the circulating supply
                               is computed from the simulation start date and pre-seeded with
                               historical data for dates prior to the simulation start date.
                               In backtesting, it was observed that compute_cs_from_start=True
                               leads to a more accurate simulation of the circulating supply, so this
                               is the default option. This option has no effect on the power predictions, 
                               only circulating supply.
        use_historical_gas: if True, gas prices are seeded from historical data. If False, gas prices
                            are computed as a constant average value. In backtesting, it was observed
                            that use_historical_gas=False leads to better backtesting results. This option
                            is only relevant when compute_cs_from_start=True.
        price_process_kwargs: a dictionary of keyword arguments to pass to the price process
        minting_process_kwargs: a dictionary of keyword arguments to pass to the minting process
        capital_inflow_process_kwargs: a dictionary of keyword arguments to pass to the capital inflow process
        capital_inflow_distribution_policy: a function that determines the percentage of total capital inflow that is allocated to a given miner
        capital_inflow_distribution_policy_kwargs: a dictionary of keyword arguments to pass to the capital inflow distribution policy
        """
        self.num_agents = n
        self.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT = constants.MAX_DAY_ONBOARD_RBP_PIB / n
        
        # TODO: I think these should become configuration objects, this is getting a bit wary ... 
        self.price_process_kwargs = price_process_kwargs
        if self.price_process_kwargs is None:
            self.price_process_kwargs = {}
        self.minting_process_kwargs = minting_process_kwargs
        if self.minting_process_kwargs is None:
            self.minting_process_kwargs = {}
        self.capital_inflow_process_kwargs = capital_inflow_process_kwargs
        if self.capital_inflow_process_kwargs is None:
            self.capital_inflow_process_kwargs = {}
        self.capital_inflow_distribution_policy = capital_inflow_distribution_policy
        if self.capital_inflow_distribution_policy is None:
            self.capital_inflow_distribution_policy = capital_inflow_process.power_proportional_capital_distribution_policy
        self.capital_inflow_distribution_policy_kwargs = capital_inflow_distribution_policy_kwargs
        if self.capital_inflow_distribution_policy_kwargs is None:
            self.capital_inflow_distribution_policy_kwargs = {}

        self.random_seed = random_seed
        self.schedule = mesa.time.SimultaneousActivation(self)

        if agent_power_distributions is None:
            self.agent_power_distributions = distribute_agent_power_geometric_series(n)
        else:
            self.agent_power_distributions = agent_power_distributions

        self.compute_cs_from_networkdatastart = compute_cs_from_networkdatastart
        self.use_historical_gas = use_historical_gas

        if not compute_cs_from_networkdatastart:
            raise ValueError("Value only True supported for now ...")

        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date
        self.sim_len = (self.end_date - self.start_date).days

        self.start_day = (self.start_date - constants.NETWORK_DATA_START).days
        self.current_day = (self.current_date - constants.NETWORK_DATA_START).days

        self.agents = []
        self.rbp0 = None
        self.qap0 = None

        self._validate(agent_kwargs_list)

        self._initialize_network_description_df()
        self._download_historical_data()
        self._seed_agents(agent_types=agent_types, agent_kwargs_list=agent_kwargs_list)
        self._fast_forward_to_simulation_start()

        self._setup_global_forecasts()

    def step(self):
        # update global forecasts
        self._update_global_forecasts()

        # step agents
        self.schedule.step()

        day_macro_info = self._compute_macro(self.current_date)
        self._update_power_metrics(day_macro_info)
        self._update_minting()
        self._update_sched_expire_pledge(self.current_date)
        self._update_circulating_supply()
        self._update_generated_quantities()

        self._step_post_network_updates()

        self._update_agents()
        # update any other inputs to agents

        # increment counters
        self.current_date += timedelta(days=1)
        self.current_day += 1

    def _setup_global_forecasts(self):
        self.global_forecast_df = pd.DataFrame()
        self.global_forecast_df['date'] = self.filecoin_df['date']

        # need to forecast this many days after the simulation end date because
        # agents will be making decisions uptil the end of simulation with future forecasts
        final_date = self.filecoin_df['date'].iloc[-1]
        remaining_len = constants.MAX_SECTOR_DURATION_DAYS
        future_dates = [final_date + timedelta(days=i) for i in range(1, remaining_len + 1)]
        self.global_forecast_df = pd.concat([self.global_forecast_df, pd.DataFrame({'date': future_dates})], ignore_index=True)
        
        self.price_process = price_process.PriceProcess(self, **self.price_process_kwargs)
        self.minting_process = rewards_per_sector_process.RewardsPerSectorProcess(self, **self.minting_process_kwargs)
        self.capital_inflow_process = capital_inflow_process.CapitalInflowProcess(self, **self.capital_inflow_process_kwargs)

    def _update_global_forecasts(self):
        # call stuff here that should be updated before agents make decisions
        self.price_process.step()
        self.minting_process.step()
    
    def _step_post_network_updates(self):
        # call stuff here that should be run after all network statistics have been updated
        self.capital_inflow_process.step()

    def estimate_pledge_for_qa_power(self, date_in, qa_power_pib):
        """
        Computes the required pledge for a given date and desired QA power to onboard.
        The general use-case for this function will be that for a given day, the agent
        is deciding whether to onboard a certain amount of power. The agent can call
        this function to determine the required pledge to onboard that amount of power,
        and then make a decision.

        Note that in the step function above, the agent is first called to make a decision.
        After all agents have made a decisions, the model aggregates the decisions for that 
        day and computes econometrics that depend on the agent's decisions, such as the total
        network QAP, circulating supply, etc. So, for a given time t, this function uses econometrics
        from time t-1 to estimate the pledge requirement.

        If the agent decides to pledge power, the actual required pledge is computed after all global
        metrics are computed and this is logged in the agent's accounting_df dataframe.

        Parameters
        ----------
        date_in : datetime.date
            date for which to compute the pledge
        qa_power_pib : float
            QA power to onboard
        """
        filecoin_df_idx = self.filecoin_df[self.filecoin_df['date'] == date_in].index[0]
        prev_day_idx = filecoin_df_idx - 1
        
        prev_circ_supply = self.filecoin_df.loc[prev_day_idx, 'circ_supply']
        prev_total_qa_power_pib = self.filecoin_df.loc[prev_day_idx, 'total_qa_power_eib'] * 1024.0
        prev_baseline_power_pib = self.filecoin_df.loc[prev_day_idx, 'network_baseline'] / constants.PIB
        prev_day_network_reward = self.filecoin_df.loc[prev_day_idx, 'day_network_reward']

        # estimate 20 days block reward
        storage_pledge = 20.0 * prev_day_network_reward * (qa_power_pib / prev_total_qa_power_pib)
        # consensus collateral
        normalized_qap_growth = qa_power_pib / max(prev_total_qa_power_pib, prev_baseline_power_pib)
        consensus_pledge = max(self.lock_target * prev_circ_supply * normalized_qap_growth, 0)
        # total added pledge
        added_pledge = storage_pledge + consensus_pledge

        pledge_cap = qa_power_pib * 1.0 / constants.GIB
        return min(pledge_cap, added_pledge)

    def _validate(self, agent_kwargs_list):
        if self.start_date < constants.NETWORK_DATA_START:
            raise ValueError(f"start_date must be after {constants.NETWORK_DATA_START}")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be after start_date")
        assert len(self.agent_power_distributions) == self.num_agents
        assert np.isclose(sum(self.agent_power_distributions), 1.0)

        assert len(self.agent_power_distributions) == self.num_agents
        if agent_kwargs_list is not None:
            assert len(agent_kwargs_list) == self.num_agents

    def _download_historical_data(self):
        # TODO: have an offline method to speed this up ... otherwise takes 30s to initialize the model
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
        self.df_historical = merged_df[
            [
                'date', 
                'day_onboarded_rb_power_pib', 'extended_rb', 'total_rb', 'terminated_rb',
                'day_onboarded_qa_power_pib', 'extended_qa', 'total_qa', 'terminated_qa',
            ]
        ]
        # rename columns for internal consistency
        self.df_historical = self.df_historical.rename(
            columns={
                'extended_rb': 'extended_rb_pib',
                'extended_qa': 'extended_qa_pib',
                'total_rb': 'sched_expire_rb_pib',
                'total_qa': 'sched_expire_qa_pib',
                'terminated_rb': 'terminated_rb_pib',
                'terminated_qa': 'terminated_qa_pib',
            }
        )
        scheduled_df = scheduled_df.rename(
            columns={
                'total_rb': 'sched_expire_rb_pib',
                'total_qa': 'sched_expire_qa_pib',
            }
        )

        final_date_historical = historical_stats.iloc[-1]['date']
        self.df_future = scheduled_df[scheduled_df['date'] >= final_date_historical][['date', 'sched_expire_rb_pib', 'sched_expire_qa_pib']]

        self.rbp0 = merged_df.iloc[0]['total_raw_power_eib']
        self.qap0 = merged_df.iloc[0]['total_qa_power_eib']
        self.max_date_se_power = self.df_future.iloc[-1]['date']
        
        # this vector starts from the first day of the simulation
        # len_since_network_start = (self.end_date - constants.NETWORK_DATA_START).days
        known_scheduled_pledge_release_vec = scheduled_df["total_pledge"].values
        start_idx = self.filecoin_df[self.filecoin_df['date'] == scheduled_df.iloc[0]['date']].index[0]
        end_idx = self.filecoin_df[self.filecoin_df['date'] == scheduled_df.iloc[-1]['date']].index[0]
        self.filecoin_df['scheduled_pledge_release'] = 0
        self.filecoin_df.loc[start_idx:end_idx, 'scheduled_pledge_release'] = known_scheduled_pledge_release_vec
        
        self.lock_target = 0.3

        self.zero_cum_capped_power = data.get_cum_capped_rb_power(constants.NETWORK_DATA_START)

    def _seed_agents(self, agent_types=None, agent_kwargs_list=None):
        for ii in range(self.num_agents):
            agent_power_pct = self.agent_power_distributions[ii]
            agent_historical_df = self.df_historical.drop('date', axis=1) * agent_power_pct
            agent_historical_df['date'] = self.df_historical['date']
            agent_future_df = self.df_future.drop('date', axis=1) * agent_power_pct
            agent_future_df['date'] = self.df_future['date']
            agent_seed = {
                'historical_power': agent_historical_df,
                'future_se_power': agent_future_df
            }
            if agent_types is not None:
                agent_cls = agent_types[ii]
            else:
                agent_cls = sp_agent.SPAgent
            
            agent_kwargs = {}
            if agent_kwargs_list is not None:
                agent_kwargs = agent_kwargs_list[ii]
            agent = agent_cls(self, ii, agent_seed, self.start_date, self.end_date, **agent_kwargs)

            self.schedule.add(agent)
            self.agents.append(
                {
                    'agent_power_pct': agent_power_pct,
                    'agent': agent,
                }
            )

    def _initialize_network_description_df(self):
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

        # generated quantities which are only updated from simulation_start
        self.filecoin_df['day_pledge_per_QAP'] = 0.0
        self.filecoin_df['day_rewards_per_sector'] = 0.0
        self.filecoin_df['capital_inflow_pct'] = 0.0
        self.filecoin_df['capital_inflow_FIL'] = 0.0
    
    def _fast_forward_to_simulation_start(self):
        current_date = constants.NETWORK_DATA_START
        day_power_stats_vec = []
        print('Fast forwarding power to simulation start date...', self.start_date)
        while current_date < self.start_date:
            day_power_stats = self._compute_macro(current_date)
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
        # TODO: better error messages
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
        remaining_power_stats_df = pd.DataFrame(np.nan, index=range(self.sim_len), columns=power_stats_df.columns)
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
        print('Computing Scheduled Expirations from: ', self.current_date, ' to: ', self.max_date_se_power)
        while current_date < self.max_date_se_power:
            se_power_stats = self._compute_macro(current_date)
            se_power_stats_vec.append(se_power_stats)

            current_date += timedelta(days=1)
        se_power_stats_df = pd.DataFrame(se_power_stats_vec)

        l = len(se_power_stats_df)
        self.filecoin_df.loc[final_historical_data_idx+1:final_historical_data_idx+l, ['day_sched_expire_rbp_pib']] = se_power_stats_df['day_sched_expire_rbp_pib'].values
        self.filecoin_df.loc[final_historical_data_idx+1:final_historical_data_idx+l, ['day_sched_expire_qap_pib']] = se_power_stats_df['day_sched_expire_qap_pib'].values

        #####################################################################################
        # initialize the circulating supply
        supply_df = data.query_supply_stats(constants.NETWORK_DATA_START, self.start_date)
        start_idx = self.filecoin_df[self.filecoin_df['date'] == supply_df.iloc[0]['date']].index[0]
        end_idx = self.filecoin_df[self.filecoin_df['date'] == supply_df.iloc[-1]['date']].index[0]
        
        self.filecoin_df['disbursed_reserve'] = (17066618961773411890063046 * 10**-18)  # constant across time.
        self.filecoin_df['network_gas_burn'] = 0

        # internal CS metrics are initialized to 0 and only filled after the simulation start date
        self.filecoin_df['day_locked_pledge'] = 0
        self.filecoin_df['day_renewed_pledge'] = 0
        self.filecoin_df['network_locked_pledge'] = 0
        self.filecoin_df['network_locked_reward'] = 0
        self.daily_burnt_fil = supply_df["burnt_fil"].diff().mean()

        # test consistency between mechaFIL and agentFIL by computing CS from beginning of simulation
        # rather than after simulation start only
        if self.compute_cs_from_networkdatastart:
            if self.use_historical_gas:
                self.filecoin_df.loc[start_idx:end_idx, 'network_gas_burn'] = supply_df['burnt_fil'].values
            locked_fil_zero = supply_df.iloc[start_idx]['locked_fil']
            self.filecoin_df.loc[start_idx, "network_locked_pledge"] = locked_fil_zero / 2.0
            self.filecoin_df.loc[start_idx, "network_locked_reward"] = locked_fil_zero / 2.0
            self.filecoin_df.loc[start_idx, "network_locked"] = locked_fil_zero
            self.filecoin_df.loc[start_idx, 'circ_supply'] = supply_df.iloc[start_idx]['circulating_fil']
            for day_idx in range(start_idx+1, end_idx):
                date_in = self.filecoin_df.loc[day_idx, 'date']
                self._update_sched_expire_pledge(date_in, update_filecoin_df=False)
                self._update_circulating_supply(update_day=day_idx)
                self._update_generated_quantities(update_day=day_idx)
                self._update_agents(update_day=day_idx)
        else:
            # NOTE: cum_network_reward was computed above from power inputs, use that rather than historical data
            # NOTE: vesting was computed above and is a static model, so use the precomputed vesting information
            # self.filecoin_df.loc[start_idx:end_idx, 'total_vest'] = supply_df['vested_fil'].values
            self.filecoin_df.loc[start_idx:end_idx, 'network_locked'] = supply_df['locked_fil'].values
            self.filecoin_df.loc[start_idx:end_idx, 'network_gas_burn'] = supply_df['burnt_fil'].values
            
            # compute circulating supply rather than overwriting it with historical data to be consistent
            # with minting model
            self.filecoin_df.loc[start_idx:end_idx, 'circ_supply'] = (
                self.filecoin_df.loc[start_idx:end_idx, 'disbursed_reserve']
                + self.filecoin_df.loc[start_idx:end_idx, 'cum_network_reward']  # from the minting_model
                + self.filecoin_df.loc[start_idx:end_idx, 'total_vest']  # from vesting_model
                - self.filecoin_df.loc[start_idx:end_idx, 'network_locked']  # from simulation loop
                - self.filecoin_df.loc[start_idx:end_idx, 'network_gas_burn']  # comes from user inputs
            )
            locked_fil_zero = self.filecoin_df.loc[final_historical_data_idx, ["network_locked"]].values[0]
            self.filecoin_df.loc[final_historical_data_idx+1, ["network_locked_pledge"]] = locked_fil_zero / 2.0
            self.filecoin_df.loc[final_historical_data_idx+1, ["network_locked_reward"]] = locked_fil_zero / 2.0
            self.filecoin_df.loc[final_historical_data_idx+1, ["network_locked"]] = locked_fil_zero

        ############################################################################################################
        day_onboarded_power_QAP = self.filecoin_df.loc[day_idx, "day_onboarded_qap_pib"] * constants.PIB   # in bytes
        network_QAP = self.filecoin_df["total_qa_power_eib"] * constants.EIB                  # in bytes

        # NOTE: this only works if compute_cs_from_networkdatastart is set to True
        self.filecoin_df['day_pledge_per_QAP'] = constants.SECTOR_SIZE * (self.filecoin_df['day_locked_pledge']-self.filecoin_df['day_renewed_pledge'])/day_onboarded_power_QAP
        self.filecoin_df['day_rewards_per_sector'] = constants.SECTOR_SIZE * self.filecoin_df['day_network_reward'] / network_QAP
        
    def _compute_macro(self, date_in):
        # compute Macro econometrics (Power statistics)
        total_rb_delta, total_qa_delta = 0, 0
        total_onboarded_rb_delta, total_renewed_rb_delta, total_se_rb_delta, total_terminated_rb_delta = 0, 0, 0, 0
        total_onboarded_qa_delta, total_renewed_qa_delta, total_se_qa_delta, total_terminated_qa_delta = 0, 0, 0, 0
        for agent_info in self.agents:
            agent = agent_info['agent']
            agent_day_power_stats = agent.get_power_at_date(date_in)
            
            total_onboarded_rb_delta += agent_day_power_stats['day_onboarded_rb_power_pib']
            total_onboarded_qa_delta += agent_day_power_stats['day_onboarded_qa_power_pib']
            
            total_renewed_rb_delta += agent_day_power_stats['extended_rb_pib']
            total_renewed_qa_delta += agent_day_power_stats['extended_qa_pib']

            total_se_rb_delta += agent_day_power_stats['sched_expire_rb_pib']
            total_se_qa_delta += agent_day_power_stats['sched_expire_qa_pib']

            total_terminated_rb_delta += agent_day_power_stats['terminated_rb_pib']
            total_terminated_qa_delta += agent_day_power_stats['terminated_qa_pib']

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
        }
        return out_dict

    def _update_power_metrics(self, day_macro_info, day_idx=None):
        day_idx = self.current_day if day_idx is None else day_idx

        self.filecoin_df.loc[day_idx, 'day_onboarded_rbp_pib'] = day_macro_info['day_onboarded_rbp_pib']
        ## FLAG: div / 0 protection
        self.filecoin_df.loc[day_idx, 'day_onboarded_qap_pib'] = max(day_macro_info['day_onboarded_qap_pib'], constants.MIN_VALUE)
        self.filecoin_df.loc[day_idx, 'day_renewed_rbp_pib'] = day_macro_info['day_renewed_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_renewed_qap_pib'] = day_macro_info['day_renewed_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_sched_expire_rbp_pib'] = day_macro_info['day_sched_expire_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_sched_expire_qap_pib'] = day_macro_info['day_sched_expire_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_terminated_rbp_pib'] = day_macro_info['day_terminated_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_terminated_qap_pib'] = day_macro_info['day_terminated_qap_pib']
        self.filecoin_df.loc[day_idx, 'day_network_rbp_pib'] = day_macro_info['day_network_rbp_pib']
        self.filecoin_df.loc[day_idx, 'day_network_qap_pib'] = day_macro_info['day_network_qap_pib']

        ## FLAG: div / 0 protection
        self.filecoin_df.loc[day_idx, 'total_raw_power_eib'] = max(self.filecoin_df.loc[day_idx, 'day_network_rbp_pib'] / 1024.0 + self.filecoin_df.loc[day_idx-1, 'total_raw_power_eib'], constants.MIN_VALUE)
        self.filecoin_df.loc[day_idx, 'total_qa_power_eib'] = max(self.filecoin_df.loc[day_idx, 'day_network_qap_pib'] / 1024.0 + self.filecoin_df.loc[day_idx-1, 'total_qa_power_eib'], constants.MIN_VALUE)

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

    def _update_sched_expire_pledge(self, date_in, update_filecoin_df=True):
        """
        Update the scheduled pledge release for the day. Track this for each agent.

        Parameters
        ----------
        date_in : str
            date to update
        update_filecoin_df : bool
            whether to update the filecoin_df with the new information.  This should be set to
            True for all user cases. In the special case where we are fast-forwarding the
            simulation, we set it to False because the relevant information was already updated.
        """
        day_idx = self.filecoin_df[self.filecoin_df['date'] == date_in].index[0]
        
        total_qa = self.filecoin_df.loc[day_idx, "total_qa_power_eib"] * constants.EIB
        baseline_power = self.filecoin_df.loc[day_idx, "network_baseline"]
        day_network_reward = self.filecoin_df.loc[day_idx, "day_network_reward"]
        prev_circ_supply = self.filecoin_df.loc[day_idx-1, "circ_supply"]
        
        for agent_info in self.agents:
            agent = agent_info['agent']
            agent_day_power_stats = agent.get_power_at_date(date_in)
        
            day_onboarded_qap = agent_day_power_stats['day_onboarded_qa_power_pib'] * constants.PIB
            day_renewed_qap = agent_day_power_stats['extended_qa_pib'] * constants.PIB
            
            # compute total pledge this agent will locked
            onboards_locked = locking.compute_new_pledge_for_added_power(
                day_network_reward,
                prev_circ_supply,
                day_onboarded_qap,
                total_qa,
                baseline_power,
                self.lock_target,
            )
            renews_locked = locking.compute_new_pledge_for_added_power(
                day_network_reward,
                prev_circ_supply,
                day_renewed_qap,
                total_qa,
                baseline_power,
                self.lock_target,
            )
            # NOTE: we can store this information back into the agent so that the agent knows
            # how much pledge it has locked. That can be part of the agent's calculation for 
            # how to proceed, based on its limited capital.

            # get the original pledge that was scheduled to expire on this day
            original_pledge = agent.accounting_df.loc[day_idx, "renew_scheduled_pledge_release_FIL"]
            renews_locked = max(original_pledge, renews_locked) if day_renewed_qap > 0 else 0

            onboarded_qa_duration = agent_day_power_stats['day_onboarded_qa_duration']
            renewed_qa_duration = agent_day_power_stats['extended_qa_duration']
            
            # only update the vector if it is within the simulation range
            agent.accounting_df.loc[day_idx, "onboard_pledge_FIL"] += onboards_locked
            if day_idx + onboarded_qa_duration < len(self.filecoin_df):
                if update_filecoin_df:
                    self.filecoin_df.loc[day_idx + onboarded_qa_duration, "scheduled_pledge_release"] += onboards_locked
                agent.accounting_df.loc[day_idx + onboarded_qa_duration, "onboard_scheduled_pledge_release_FIL"] += onboards_locked

            agent.accounting_df.loc[day_idx, "renew_pledge_FIL"] += renews_locked
            if day_idx + renewed_qa_duration < len(self.filecoin_df):
                if update_filecoin_df:
                    self.filecoin_df.loc[day_idx + renewed_qa_duration, "scheduled_pledge_release"] += renews_locked
                agent.accounting_df.loc[day_idx + renewed_qa_duration, "renew_scheduled_pledge_release_FIL"] += renews_locked
            
    def _update_circulating_supply(self, update_day=None):
        day_idx = self.current_day if update_day is None else update_day

        # day_pledge_locked_vec = self.filecoin_df["day_locked_pledge"].values
        day_onboarded_power_QAP = self.filecoin_df.iloc[day_idx]["day_onboarded_qap_pib"] * constants.PIB   # in bytes
        day_renewed_power_QAP = self.filecoin_df.iloc[day_idx]["day_renewed_qap_pib"] * constants.PIB       # in bytes
        network_QAP = self.filecoin_df.iloc[day_idx]["total_qa_power_eib"] * constants.EIB                  # in bytes
        network_baseline = self.filecoin_df.iloc[day_idx]["network_baseline"]                               # in bytes
        day_network_reward = self.filecoin_df.iloc[day_idx]["day_network_reward"]
        
        day_sched_expire_rbp_pib = self.filecoin_df.iloc[day_idx]["day_sched_expire_rbp_pib"]
        if day_sched_expire_rbp_pib == 0:
            renewal_rate = 0.0
        else:
            renewal_rate = self.filecoin_df.iloc[day_idx]["day_renewed_rbp_pib"] / day_sched_expire_rbp_pib

        prev_network_locked_reward = self.filecoin_df.iloc[day_idx-1]["network_locked_reward"]
        prev_network_locked_pledge = self.filecoin_df.iloc[day_idx-1]["network_locked_pledge"]
        prev_network_locked = self.filecoin_df.iloc[day_idx-1]["network_locked"]

        prev_circ_supply = self.filecoin_df["circ_supply"].iloc[day_idx-1]

        # scheduled_pledge_release = supply.get_day_schedule_pledge_release(
        #     day_idx,
        #     self.current_day,
        #     day_pledge_locked_vec,
        #     self.known_scheduled_pledge_release_full_vec,
        #     self.sector_duration,
        # )
        scheduled_pledge_release = self.filecoin_df["scheduled_pledge_release"].iloc[day_idx]
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
        if self.filecoin_df.loc[day_idx, "network_gas_burn"] == 0.0:
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

    def _update_generated_quantities(self, update_day=None):
        day_idx = self.current_day if update_day is None else update_day

        # add ROI to trajectory df
        day_locked_pledge = self.filecoin_df.loc[day_idx, 'day_locked_pledge']
        day_renewed_pledge = self.filecoin_df.loc[day_idx, 'day_renewed_pledge']
        # FLAG: avoid division by zero - does it make sense to do this?
        day_onboarded_power_QAP = max(self.filecoin_df.loc[day_idx, "day_onboarded_qap_pib"] * constants.PIB, constants.MIN_VALUE)   # in bytes
        self.filecoin_df.loc[day_idx, 'day_pledge_per_QAP'] = constants.SECTOR_SIZE * (day_locked_pledge-day_renewed_pledge)/day_onboarded_power_QAP

        day_network_reward = self.filecoin_df.iloc[day_idx]["day_network_reward"]
        # FLAG: avoid division by zero - does it make sense to do this?
        network_QAP = max(self.filecoin_df.iloc[day_idx]["total_qa_power_eib"] * constants.EIB, constants.MIN_VALUE)                  # in bytes
        self.filecoin_df.loc[day_idx, 'day_rewards_per_sector'] = constants.SECTOR_SIZE * day_network_reward / network_QAP
        
        
    def _update_agents(self, update_day=None):
        day_idx = self.current_day if update_day is None else update_day
        date_in = self.filecoin_df.iloc[day_idx]['date']

        total_day_rewards = self.filecoin_df.iloc[day_idx]["day_network_reward"]
        total_day_onboard_and_renew_pib = self.filecoin_df.iloc[day_idx]["day_onboarded_qap_pib"] +  self.filecoin_df.iloc[day_idx]["day_renewed_qap_pib"]

        # TODO: add termination penalties here

        total_day_capital_inflow_FIL = self.filecoin_df.loc[day_idx, 'capital_inflow_FIL']
        for agent_info in self.agents:
            agent = agent_info['agent']
            agent_day_power_stats = agent.get_power_at_date(date_in)
        
            day_onboarded_qap = agent_day_power_stats['day_onboarded_qa_power_pib']
            day_renewed_qap = agent_day_power_stats['extended_qa_pib']
            total_agent_qap_onboarded = day_onboarded_qap + day_renewed_qap
            agent_reward_ratio = min(total_agent_qap_onboarded/total_day_onboard_and_renew_pib, 1.0) # account for numerical issues
            # print(agent.unique_id, date_in, day_onboarded_qap, day_renewed_qap, total_day_onboard_and_renew_pib)
            agent_reward = total_day_rewards * agent_reward_ratio

            # agent_network_df = agent_info['network_updates_df']
            agent_accounting_df = agent.accounting_df
            accounting_df_idx = agent_accounting_df[agent_accounting_df['date'] == date_in].index[0]

            # 25 % vests immediately
            agent_accounting_df.loc[accounting_df_idx, 'reward_FIL'] += agent_reward * 0.25
            # remainder vests linearly over the next 180 days
            agent_accounting_df.loc[accounting_df_idx+1:accounting_df_idx+180, 'reward_FIL'] += (agent_reward * 0.75)/180

            # if there is any capital inflow FIL, distribute it to the agents according to the inflow distribution policy
            if total_day_capital_inflow_FIL > 0.0:
                # TODO: need to figure out how to generalize the inputs to the distribution inflow policy function
                FIL_to_agent = math.floor(self.capital_inflow_distribution_policy(total_agent_qap_onboarded, 
                                                                                  total_day_onboard_and_renew_pib, 
                                                                                  total_day_capital_inflow_FIL))
                agent.accounting_df.loc[accounting_df_idx, 'capital_inflow_FIL'] += FIL_to_agent

            agent.post_global_step()
