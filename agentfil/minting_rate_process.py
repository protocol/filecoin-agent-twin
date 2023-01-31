from datetime import timedelta

import numpy as np
import pandas as pd
import jax.numpy as jnp

from mechafil import minting
from scenario_generator import mcmc_forecast

from . import constants

class MintingRateProcess:
    def __init__(self, filecoin_model, 
                 forecast_history=180, update_every_days=90,
                 num_warmup_mcmc: int = 500,
                 num_samples_mcmc: int = 100,
                 seasonality_mcmc: int = 1000,
                 num_chains_mcmc: int = 2,
                 verbose: bool = True,
                 keep_previous_predictions: bool = False,
                 keep_rbp_predictions: bool = False):
        """
        Predicts the future minting rate by using MCMC to forecast the total Raw Byte Onboarding Power / day
        and then using the mechanistic equations to convert that to the minting rate.

        Parameters
        ----------
        filecoin_model : FilecoinModel
            The Filecoin model to which this process is attached.
        forecast_history : int, optional
            Number of days of historical data to use for forecasting, by default 180
        update_every_days : int, optional
            Number of days between forecast updates, by default 90
        num_warmup_mcmc : int, optional
            Number of warmup steps for MCMC, by default 500
        num_samples_mcmc : int, optional
            Number of samples for MCMC, by default 100
        seasonality_mcmc : int, optional
            Seasonality for MCMC, by default 1000
        num_chains_mcmc : int, optional
            Number of MCMC chains, by default 2
        verbose : bool, optional
            Whether to print MCMC progress, by default True
        keep_previous_predictions : bool, optional
            Whether to keep previous predictions when updating, by default False.  Warning,
            setting this true could eat up a lot of memory!
        keep_rbp_predictions : bool, optional
            Whether to keep raw byte power predictions, by default False.  Warning,
            setting this true could eat up a lot of memory!
        """
        # setup prediction options
        self.num_warmup_mcmc = num_warmup_mcmc
        self.num_samples_mcmc = num_samples_mcmc
        self.seasonality_mcmc = seasonality_mcmc
        self.num_chains_mcmc = num_chains_mcmc
        self.verbose = verbose
        self.keep_previous_predictions = keep_previous_predictions
        self.keep_rbp_predictions = keep_rbp_predictions

        self.model = filecoin_model

        start_date = self.model.start_date
        self.forecast_history = forecast_history
        self.update_dates = [start_date + timedelta(days=i) for i in range(0, self.model.sim_len, update_every_days)]

        self.minting_rate_from_start = minting.compute_baseline_power_array(constants.NETWORK_DATA_START, self.model.end_date + timedelta(days=constants.MAX_SECTOR_DURATION_DAYS))


    def step(self):
        ### called every tick
        if self.model.current_date in self.update_dates:
            # get relevant historical data
            historical_rb_onboard = self.model.filecoin_df['day_onboarded_rbp_pib'][self.model.current_day - self.forecast_history : self.model.current_day].values

            total_raw_power_eib_start = self.model.filecoin_df['total_raw_power_eib'][self.model.current_day - 1]
            cum_capped_power_start = self.model.filecoin_df['cum_capped_power'][self.model.current_day - 1]
            # train MCMC forecaster
            y_train = jnp.array(historical_rb_onboard)
            forecast_length = (self.model.end_date - self.model.current_date).days + constants.MAX_SECTOR_DURATION_DAYS
            y_scale = 1.0
            rb_onboard_forecast_pib = mcmc_forecast.mcmc_predict(y_train/y_scale, forecast_length,
                                                                 num_warmup_mcmc=self.num_warmup_mcmc, 
                                                                 num_samples_mcmc=self.num_samples_mcmc,
                                                                 seasonality_mcmc=self.seasonality_mcmc, 
                                                                 num_chains_mcmc=self.num_chains_mcmc,
                                                                 verbose=self.verbose)
            rb_onboard_forecast_pib = rb_onboard_forecast_pib * y_scale

            minting_df_future = self.model.filecoin_df[self.model.current_day:][['days', 'date', 'network_baseline', 'cum_simple_reward']]
            ## Generate forecast beyond end of simulation since agents will need this information when making decisions close to
            ## the end of the simulation
            minting_df_max_sector_duration_days = pd.DataFrame()
            minting_df_max_sector_duration_days['date'] = [minting_df_future['date'].values[-1] + timedelta(days=i) for i in range(1, constants.MAX_SECTOR_DURATION_DAYS + 1)]
            minting_df_max_sector_duration_days['network_baseline'] = self.minting_rate_from_start[-len(minting_df_max_sector_duration_days):]
            start_day = minting_df_future['days'].values[-1] + 1
            minting_df_max_sector_duration_days['days'] = range(start_day, start_day + constants.MAX_SECTOR_DURATION_DAYS)
            minting_df_max_sector_duration_days['cum_simple_reward'] = minting_df_max_sector_duration_days['days'].pipe(minting.cum_simple_minting)

            # print(minting_df_future.iloc[-1][['days', 'date', 'network_baseline', 'cum_simple_reward']])
            # print(minting_df_max_sector_duration_days.iloc[0][['days', 'date', 'network_baseline', 'cum_simple_reward']])

            minting_df_future = pd.concat([minting_df_future, minting_df_max_sector_duration_days], ignore_index=True)

            # update minting rate forecasts for each mcmc path
            num_mcmc = self.num_chains_mcmc * self.num_samples_mcmc
            
            rb_onboard_pred = []
            day_network_rewards_pred = []
            for i in range(num_mcmc):
                rb_onboard_forecast_pib_i = np.asarray(rb_onboard_forecast_pib[i, :])
                total_raw_power_eib = total_raw_power_eib_start + np.cumsum(rb_onboard_forecast_pib_i) / 1024.0
                
                capped_power = np.minimum(constants.EIB * total_raw_power_eib, minting_df_future['network_baseline'].values)
                cum_capped_power = np.cumsum(capped_power) + cum_capped_power_start
                network_time = minting.network_time(cum_capped_power)
                cum_baseline_reward = minting.cum_baseline_reward(network_time)
                cum_simple_reward = minting_df_future['cum_simple_reward'].values
                cum_total_reward = cum_baseline_reward + cum_simple_reward
                day_network_reward_i = np.diff(cum_total_reward)
                
                if self.keep_rbp_predictions:
                    rb_onboard_pred.append(rb_onboard_forecast_pib_i)
                day_network_rewards_pred.append(day_network_reward_i)

            # compute quantiles
            date_str = self.model.current_date.strftime('%Y-%m-%d')
            day_network_rewards_pred = np.asarray(day_network_rewards_pred)
            day_network_reward_quantiles = np.quantile(day_network_rewards_pred, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

            # update the model's dataframe for predictions
            q5 = day_network_reward_quantiles[0]; q5 = np.concatenate([q5, [q5[-1]]])  # repeat the last value to match the length of the dataframe
            q25 = day_network_reward_quantiles[1]; q25 = np.concatenate([q25, [q25[-1]]])  # repeat the last value to match the length of the dataframe
            q50 = day_network_reward_quantiles[2]; q50 = np.concatenate([q50, [q50[-1]]])  # repeat the last value to match the length of the dataframe
            q75 = day_network_reward_quantiles[3]; q75 = np.concatenate([q75, [q75[-1]]])  # repeat the last value to match the length of the dataframe
            q95 = day_network_reward_quantiles[4]; q95 = np.concatenate([q95, [q95[-1]]])  # repeat the last value to match the length of the dataframe

            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q05'] = q5
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q25'] = q25
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q50'] = q50
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q75'] = q75
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q95'] = q95

            if self.keep_previous_predictions:
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q05_%s' % (date_str,)] = q5
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q25_%s' % (date_str,)] = q25
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q50_%s' % (date_str,)] = q50
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q75_%s' % (date_str,)] = q75
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q95_%s' % (date_str,)] = q95

            # should only be used if need to debug minting rate predictions
            if self.keep_rbp_predictions:
                rb_onboard_pred = np.asarray(rb_onboard_pred)
                rb_onboard_quantiles = np.quantile(rb_onboard_pred, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)

                q5_rbp = rb_onboard_quantiles[0] 
                q25_rbp = rb_onboard_quantiles[1]
                q50_rbp = rb_onboard_quantiles[2]
                q75_rbp = rb_onboard_quantiles[3]
                q95_rbp = rb_onboard_quantiles[4]

                self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q05'] = q5_rbp
                self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q25'] = q25_rbp
                self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q50'] = q50_rbp
                self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q75'] = q75_rbp
                self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q95'] = q95_rbp

                if self.keep_previous_predictions:
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_forecast_Q5_%s' % (date_str,)] = q5_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_forecast_Q25_%s' % (date_str,)] = q25_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_forecast_Q50_%s' % (date_str,)] = q50_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_forecast_Q75_%s' % (date_str,)] = q75_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_forecast_Q95_%s' % (date_str,)] = q95_rbp
