from datetime import timedelta

import numpy as np
import pandas as pd
import jax.numpy as jnp

from mechafil import minting
from scenario_generator import mcmc_forecast

from . import constants

class RewardsPerSectorProcess:
    """
    Predicts the rewards/sector by:
        1) using MCMC to forecast Raw-Byte Onboarding Power / day and then using the 
        mechanistic equations to convert that to the minting rate.
        2) using MCMC to forecast QA onboarding power / day, and combining that
        with the minting rate prediction to get the rewards/sector.
    """
    def __init__(self, filecoin_model, 
                 forecast_history=180, update_every_days=90,
                 num_warmup_mcmc: int = 500,
                 num_samples_mcmc: int = 100,
                 seasonality_mcmc: int = 1000,
                 num_chains_mcmc: int = 2,
                 verbose: bool = True,
                 keep_previous_predictions: bool = False,
                 keep_power_predictions: bool = False,
                 seed=1234):
        """
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
        keep_power_predictions : bool, optional
            Whether to keep power predictions, by default False.  Warning,
            setting this true could eat up a lot of memory!
        """
        # setup prediction options
        self.num_warmup_mcmc = num_warmup_mcmc
        self.num_samples_mcmc = num_samples_mcmc
        self.seasonality_mcmc = seasonality_mcmc
        self.num_chains_mcmc = num_chains_mcmc
        self.verbose = verbose
        self.keep_previous_predictions = keep_previous_predictions
        self.keep_power_predictions = keep_power_predictions
        self.random_state = np.random.RandomState(seed)

        self.model = filecoin_model

        start_date = self.model.start_date
        self.forecast_history = forecast_history
        self.update_dates = [start_date + timedelta(days=i) for i in range(0, self.model.sim_len, update_every_days)]

        self.minting_rate_from_start = minting.compute_baseline_power_array(constants.NETWORK_DATA_START, self.model.end_date + timedelta(days=constants.MAX_SECTOR_DURATION_DAYS))

    def dither(self, x, range_pct=0.01):
        zero_noise_add = self.random_state.beta(1,1, size=(len(x)))  # use beta distribution so we never go negative, since this is used for power prediction primarily
        min_val = np.min(x)
        max_val = np.max(x)
        range = max_val - min_val
        x = x + (zero_noise_add * range * range_pct)  # dither 3% of the range
        return x

    def step(self):
        ### called every tick
        if self.model.current_date in self.update_dates:
            forecast_length = (self.model.end_date - self.model.current_date).days + constants.MAX_SECTOR_DURATION_DAYS

            total_raw_power_eib_start = self.model.filecoin_df['total_raw_power_eib'][self.model.current_day - 1]
            total_qa_power_eib_start = self.model.filecoin_df['total_qa_power_eib'][self.model.current_day - 1]
            cum_capped_power_start = self.model.filecoin_df['cum_capped_power'][self.model.current_day - 1]

            # get relevant historical data
            historical_rb_onboard = self.model.filecoin_df['day_onboarded_rbp_pib'][self.model.current_day - self.forecast_history : self.model.current_day].values
            historical_qa_onboard = self.model.filecoin_df['day_onboarded_qap_pib'][self.model.current_day - self.forecast_history : self.model.current_day].values

            # ##################################################################################################
            # create a simplistic extension of average historical power to the future
            # adding this in until we can diagnose the nans from MCMC
            num_mc = self.num_chains_mcmc * self.num_samples_mcmc
            deviation_pct = 0.25
            rb_onboard_forecast_pib = np.mean(historical_rb_onboard) + self.random_state.normal(0, np.std(historical_rb_onboard) * deviation_pct, size=(num_mc, forecast_length))
            qa_onboard_forecast_pib = np.mean(historical_qa_onboard) + self.random_state.normal(0, np.std(historical_qa_onboard) * deviation_pct, size=(num_mc, forecast_length))
            rb_onboard_forecast_pib = np.maximum(rb_onboard_forecast_pib, constants.MIN_VALUE)
            qa_onboard_forecast_pib = np.maximum(qa_onboard_forecast_pib, constants.MIN_VALUE)
            # ##################################################################################################

            # ##################################################################################################
            # ######## MCMC based Forecasting

            # # dither the input to prevent NANs from MCMC
            # historical_rb_onboard = self.dither(historical_rb_onboard, range_pct=0.1)
            # historical_qa_onboard = self.dither(historical_qa_onboard, range_pct=0.1)

            # # forecast rb onboard
            # y_train_rb = jnp.array(historical_rb_onboard)
            # y_scale = 1.0
            # rb_onboard_forecast_pib = mcmc_forecast.mcmc_predict(y_train_rb/y_scale, forecast_length,
            #                                                      num_warmup_mcmc=self.num_warmup_mcmc, 
            #                                                      num_samples_mcmc=self.num_samples_mcmc,
            #                                                      seasonality_mcmc=self.seasonality_mcmc, 
            #                                                      num_chains_mcmc=self.num_chains_mcmc,
            #                                                      verbose=self.verbose)
            # rb_onboard_forecast_pib = rb_onboard_forecast_pib * y_scale

            # # forecast qa onboard
            # y_train_qa = jnp.array(historical_qa_onboard)
            # forecast_length = (self.model.end_date - self.model.current_date).days + constants.MAX_SECTOR_DURATION_DAYS
            # y_scale = 1.0
            # qa_onboard_forecast_pib = mcmc_forecast.mcmc_predict(y_train_qa/y_scale, forecast_length,
            #                                                      num_warmup_mcmc=self.num_warmup_mcmc, 
            #                                                      num_samples_mcmc=self.num_samples_mcmc,
            #                                                      seasonality_mcmc=self.seasonality_mcmc, 
            #                                                      num_chains_mcmc=self.num_chains_mcmc,
            #                                                      verbose=self.verbose)
            # qa_onboard_forecast_pib = qa_onboard_forecast_pib * y_scale

            # rb_pred_np = np.asarray(rb_onboard_forecast_pib)
            # qa_pred_np = np.asarray(qa_onboard_forecast_pib)
            # if np.isnan(rb_pred_np).any() or np.isnan(qa_pred_np).any():
            #     import pickle
            #     with open('nan_onboard.pkl', 'wb') as f:
            #         output_dict = {
            #             'y_train_rb': np.asarray(y_train_rb),
            #             'y_train_qa': np.asarray(y_train_qa),
            #             'forecast_length': forecast_length,
            #             'current_day': self.model.current_day,
            #             'current_date': self.model.current_date,
            #             'forecast_history': self.forecast_history,
            #             'filecoin_df': self.model.filecoin_df,
            #             'y_scale': y_scale,
            #             'num_warmup_mcmc': self.num_warmup_mcmc,
            #             'num_samples_mcmc': self.num_samples_mcmc,
            #             'seasonality_mcmc': self.seasonality_mcmc,
            #             'num_chains_mcmc': self.num_chains_mcmc,
            #             'verbose': self.verbose,
            #             'rb_onboard_forecast_pib': rb_pred_np,
            #             'qa_onboard_forecast_pib': qa_pred_np,
            #         }
            #         pickle.dump(output_dict, f)
            #     raise ValueError("NAN")

            # # ensure all predictions > 0
            # rb_onboard_forecast_pib = jnp.maximum(rb_onboard_forecast_pib, constants.MIN_VALUE)
            # qa_onboard_forecast_pib = jnp.maximum(qa_onboard_forecast_pib, constants.MIN_VALUE)

            # ##################################################################################################

            minting_df_future = self.model.filecoin_df[self.model.current_day:][['days', 'date', 'network_baseline', 'cum_simple_reward']]
            ## Generate forecast beyond end of simulation since agents will need this information when making decisions close to
            ## the end of the simulation
            minting_df_max_sector_duration_days = pd.DataFrame()
            minting_df_max_sector_duration_days['date'] = [minting_df_future['date'].values[-1] + timedelta(days=i) for i in range(1, constants.MAX_SECTOR_DURATION_DAYS + 1)]
            minting_df_max_sector_duration_days['network_baseline'] = self.minting_rate_from_start[-len(minting_df_max_sector_duration_days):]
            start_day = minting_df_future['days'].values[-1] + 1
            minting_df_max_sector_duration_days['days'] = range(start_day, start_day + constants.MAX_SECTOR_DURATION_DAYS)
            minting_df_max_sector_duration_days['cum_simple_reward'] = minting_df_max_sector_duration_days['days'].pipe(minting.cum_simple_minting)

            minting_df_future = pd.concat([minting_df_future, minting_df_max_sector_duration_days], ignore_index=True)

            # update minting rate forecasts for each mcmc path
            num_mcmc = self.num_chains_mcmc * self.num_samples_mcmc
            
            rb_onboard_pred = []
            qa_onboard_pred = []

            day_network_rewards_pred = []
            day_rewards_per_sector_pred = []
            for i in range(num_mcmc):
                rb_onboard_forecast_pib_i = np.asarray(rb_onboard_forecast_pib[i, :])
                qa_onboard_forecast_pib_i = np.asarray(qa_onboard_forecast_pib[i, :])

                # # not sure why mcmc is predicting nan's, even though it seems the inputs to MCMC are not NAN
                # if np.isnan(rb_onboard_forecast_pib_i).any() or np.isnan(qa_onboard_forecast_pib_i).any():
                #     continue

                total_raw_power_eib = total_raw_power_eib_start + np.cumsum(rb_onboard_forecast_pib_i) / 1024.0
                total_qa_power_eib = total_qa_power_eib_start + np.cumsum(qa_onboard_forecast_pib_i) / 1024.0
                total_qa_power_eib = np.asarray(total_qa_power_eib, dtype=np.longdouble)  # NOTE: this is necessary to avoid overflow when scaling EiB to Bytes
                
                # FLAG: divide by zero protection
                total_raw_power_eib = np.maximum(total_raw_power_eib, constants.MIN_VALUE)
                total_qa_power_eib = np.maximum(total_qa_power_eib, constants.MIN_VALUE)

                capped_power = np.minimum(constants.EIB * total_raw_power_eib, minting_df_future['network_baseline'].values)
                cum_capped_power = np.cumsum(capped_power) + cum_capped_power_start
                network_time = minting.network_time(cum_capped_power)
                cum_baseline_reward = minting.cum_baseline_reward(network_time)
                cum_simple_reward = minting_df_future['cum_simple_reward'].values
                cum_total_reward = cum_baseline_reward + cum_simple_reward
                day_network_reward_i = np.diff(cum_total_reward)
                # repeat the last value to match the length of prediction
                day_network_reward_i = np.concatenate([day_network_reward_i, [day_network_reward_i[-1]]])
                
                if self.keep_power_predictions:
                    rb_onboard_pred.append(rb_onboard_forecast_pib_i)
                    qa_onboard_pred.append(qa_onboard_forecast_pib_i)
                day_network_rewards_pred.append(day_network_reward_i)
                
                day_i_reward_per_sector = constants.SECTOR_SIZE * day_network_reward_i / (total_qa_power_eib * constants.EIB)
                day_rewards_per_sector_pred.append(day_i_reward_per_sector)
                # if np.isnan(day_i_reward_per_sector).any():
                #     print('Index i', i, 'has nan')
                #     if np.isnan(rb_onboard_forecast_pib_i).any():
                #         print('WARNING: nan in rb_onboard_forecast_pib_i')

                #     if np.isnan(total_raw_power_eib).any():
                #         print('start_power', total_raw_power_eib_start, 'WARNING: nan in total_raw_power_eib')
                    
                #     if np.isnan(minting_df_future['network_baseline'].values).any():
                #         print('WARNING: nan in minting_df_future[network_baseline]')

                #     if np.isnan(day_network_reward_i).any():
                #         print('WARNING: nan in day_network_reward_i')

                #     if np.isnan(total_qa_power_eib).any():
                #         print('WARNING: nan in total_qa_power_eib')

                #     if np.isnan(day_i_reward_per_sector).any():
                #         print('WARNING: nan in day_rewards_per_sector_pred')
                        
                #     print('WARNING: nan in day_rewards_per_sector_pred')
                #     # raise ValueError('nan in day_rewards_per_sector_pred')

            # compute quantiles
            date_str = self.model.current_date.strftime('%Y-%m-%d')
            day_network_rewards_pred = np.asarray(day_network_rewards_pred)
            day_network_reward_quantiles = np.quantile(day_network_rewards_pred, constants.MC_QUANTILES, axis=0)
            day_rewards_per_sector_quantiles = np.quantile(day_rewards_per_sector_pred, constants.MC_QUANTILES, axis=0)

            # update the model's dataframe for predictions
            day_network_reward_q5 = day_network_reward_quantiles[0]
            day_network_reward_q25 = day_network_reward_quantiles[1]
            day_network_reward_q50 = day_network_reward_quantiles[2]
            day_network_reward_q75 = day_network_reward_quantiles[3]
            day_network_reward_q95 = day_network_reward_quantiles[4]
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q05'] = day_network_reward_q5
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q25'] = day_network_reward_q25
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q50'] = day_network_reward_q50
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q75'] = day_network_reward_q75
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q95'] = day_network_reward_q95

            day_rewards_per_sector_q5 = day_rewards_per_sector_quantiles[0]
            day_rewards_per_sector_q25 = day_rewards_per_sector_quantiles[1]
            day_rewards_per_sector_q50 = day_rewards_per_sector_quantiles[2]
            day_rewards_per_sector_q75 = day_rewards_per_sector_quantiles[3]
            day_rewards_per_sector_q95 = day_rewards_per_sector_quantiles[4]
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q05'] = day_rewards_per_sector_q5
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q25'] = day_rewards_per_sector_q25
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q50'] = day_rewards_per_sector_q50
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q75'] = day_rewards_per_sector_q75
            self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q95'] = day_rewards_per_sector_q95

            if self.keep_previous_predictions:
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q05_%s' % (date_str,)] = day_network_reward_q5
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q25_%s' % (date_str,)] = day_network_reward_q25
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q50_%s' % (date_str,)] = day_network_reward_q50
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q75_%s' % (date_str,)] = day_network_reward_q75
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_network_reward_forecast_Q95_%s' % (date_str,)] = day_network_reward_q95

                self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q05_%s' % (date_str,)] = day_rewards_per_sector_q5
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q25_%s' % (date_str,)] = day_rewards_per_sector_q25
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q50_%s' % (date_str,)] = day_rewards_per_sector_q50
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q75_%s' % (date_str,)] = day_rewards_per_sector_q75
                self.model.global_forecast_df.loc[self.model.current_day:, 'day_rewards_per_sector_forecast_Q95_%s' % (date_str,)] = day_rewards_per_sector_q95

            # should only be used if need to debug minting rate predictions
            if self.keep_power_predictions:
                rb_onboard_pred = np.asarray(rb_onboard_pred)
                qa_onboard_pred = np.asarray(qa_onboard_pred)
                rb_onboard_quantiles = np.quantile(rb_onboard_pred, constants.MC_QUANTILES, axis=0)
                qa_onboard_quantiles = np.quantile(qa_onboard_pred, constants.MC_QUANTILES, axis=0)

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

                q5_qap = qa_onboard_quantiles[0] 
                q25_qap = qa_onboard_quantiles[1]
                q50_qap = qa_onboard_quantiles[2]
                q75_qap = qa_onboard_quantiles[3]
                q95_qap = qa_onboard_quantiles[4]
                self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q05'] = q5_qap
                self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q25'] = q25_qap
                self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q50'] = q50_qap
                self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q75'] = q75_qap
                self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q95'] = q95_qap

                if self.keep_previous_predictions:
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q05_%s' % (date_str,)] = q5_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q25_%s' % (date_str,)] = q25_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q50_%s' % (date_str,)] = q50_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q75_%s' % (date_str,)] = q75_rbp
                    self.model.global_forecast_df.loc[self.model.current_day:, 'rb_onboard_pib_forecast_Q95_%s' % (date_str,)] = q95_rbp

                    self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q05_%s' % (date_str,)] = q5_qap
                    self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q25_%s' % (date_str,)] = q25_qap
                    self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q50_%s' % (date_str,)] = q50_qap
                    self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q75_%s' % (date_str,)] = q75_qap
                    self.model.global_forecast_df.loc[self.model.current_day:, 'qa_onboard_pib_forecast_Q95_%s' % (date_str,)] = q95_qap
