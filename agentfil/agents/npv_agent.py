from datetime import timedelta
from .. import constants
from .sp_agent import SPAgent
from ..power import cc_power, deal_power

import numpy as np
import pandas as pd

class NPVAgent(SPAgent):
    """
    The NPV agent is an agent that uses rewards forecasts to decide how much power to onboard.
    It computes the NPV / sector and if if NPV > 0, the agent will decide to onboard the configured
    amount of power. Otherwise, it will not.  Same logic applies to renewals.

    TODO:
     [ ] - 
    """
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, max_daily_rb_onboard_pib=3,
                 renewal_rate = 0.6, fil_plus_rate=0.6, 
                 agent_optimism=4, agent_discount_rate_yr_pct=50):
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)

        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate

        self.agent_optimism = agent_optimism
        self.agent_discount_rate_yr_pct = agent_discount_rate_yr_pct
        self.agent_discount_rate_yr = self.agent_discount_rate_yr_pct / 100.

        self.duration_vec_days = np.asarray([365, 1095, 1825]).astype(np.int32)  # 1Y, 3Y, 5Y sectors are possible

        self.map_optimism_scales()

        for d in self.duration_vec_days:
            self.agent_info_df[f'npv_estimate_{d}'] = 0

    def map_optimism_scales(self):
        self.optimism_to_price_quantile_str = {
            1 : "Q05",
            2 : "Q25",
            3 : "Q50",
            4 : "Q75",
            5 : "Q95"
        }
        self.optimism_to_dayrewardspersector_quantile_str = {
            1 : "Q05",
            2 : "Q25",
            3 : "Q50",
            4 : "Q75",
            5 : "Q95"
        }

    def forecast_day_rewards_per_sector(self, forecast_start_date, forecast_length):
        k = 'day_rewards_per_sector_forecast_' + self.optimism_to_dayrewardspersector_quantile_str[self.agent_optimism]
        start_idx = self.model.global_forecast_df[pd.to_datetime(self.model.global_forecast_df['date']) == pd.to_datetime(forecast_start_date)].index[0]
        end_idx = start_idx + forecast_length
        future_rewards_per_sector = self.model.global_forecast_df.loc[start_idx:end_idx, k].values
        
        return future_rewards_per_sector


    def estimate_npv(self, sector_duration, date_in):
        sector_duration_yrs = sector_duration / 365.
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]

        # NOTE: we need to use yesterday's metrics b/c today's haven't yet been aggregated by the system yet
        prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']

        # TODO: make this an iterative update rather than full-estimate every day
        # NOTE: this assumes that the pledge remains constant. This is not true, but a zeroth-order approximation
        future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        future_rewards_estimate = np.sum(future_rewards_per_sector_estimate)
        # continuous discounting
        future_rewards_estimate_discounted = future_rewards_estimate / np.exp(self.agent_discount_rate_yr * sector_duration_yrs)
        
        # get the cost per sector for the duration, which in this case is just borrowing costs
        pledge_repayment_estimate = self.compute_repayment_amount_from_supply_discount_rate_model(date_in, prev_day_pledge_per_QAP, sector_duration_yrs)
        cost_per_sector_estimate = pledge_repayment_estimate - prev_day_pledge_per_QAP

        cost_per_sector_estimate_discounted = cost_per_sector_estimate / np.exp(self.agent_discount_rate_yr * sector_duration_yrs)
        
        npv_estimate = future_rewards_estimate_discounted - cost_per_sector_estimate_discounted
        
        return npv_estimate

    def step(self):
        npv_estimate_vec = []
        
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(self.current_date)].index[0]
        for d in self.duration_vec_days:    
            npv_estimate = self.estimate_npv(d, self.current_date)            
            npv_estimate_vec.append(npv_estimate)
            self.agent_info_df.loc[agent_df_idx, 'npv_estimate_%d' % (d,)] = npv_estimate
            
        max_npv_idx = np.argmax(npv_estimate_vec)
        best_duration = self.duration_vec_days[max_npv_idx]
        best_duration_yrs = best_duration / 365.
        if npv_estimate_vec[max_npv_idx] > 0:
            # for now, we put all power into FIL+ (deal power)
            rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
            qa_to_onboard = self.model.apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate,
                                                       fil_plus_multipler=constants.FIL_PLUS_MULTIPLER,
                                                       date_in=self.current_date,
                                                       sector_duration_days=best_duration) + \
                        rb_to_onboard * (1-self.fil_plus_rate)
            pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)

            pledge_needed_for_onboarding = qa_to_onboard * pledge_per_pib
            pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                           pledge_needed_for_onboarding, 
                                                                                                           best_duration_yrs)

            self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, best_duration,
                               pledge_needed_for_onboarding, pledge_repayment_value_onboard)

            # renew available power for the same duration
            if self.renewal_rate > 0:
                se_power_dict = self.get_se_power_at_date(self.current_date)
                # which aspects of power get renewed is dependent on the setting "renewals_setting" in the FilecoinModel object
                cc_power_to_renew = se_power_dict['se_cc_power'] * self.renewal_rate
                deal_power_to_renew = se_power_dict['se_deal_power'] * self.renewal_rate

                pledge_needed_for_renewal = (cc_power_to_renew + deal_power_to_renew) * pledge_per_pib
                pledge_repayment_value_renew = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                            pledge_needed_for_renewal, 
                                                                                                            best_duration_yrs)

                self.renew_power(self.current_date, cc_power_to_renew, deal_power_to_renew, best_duration,
                                pledge_needed_for_renewal, pledge_repayment_value_renew)

        super().step()

    def post_global_step(self):
        # we can update local representation of anything else that should happen after
        # global metrics for day are aggregated
        pass