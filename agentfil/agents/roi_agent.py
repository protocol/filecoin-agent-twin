from datetime import timedelta
from . import constants
from .sp_agent import SPAgent
from ..power import cc_power, deal_power
from ..filecoin_model import apply_qa_multiplier

import numpy as np
import pandas as pd

class ROIAgent(SPAgent):
    """
    The ROI agent is an agent that uses ROI forecasts to decide how much power to onboard.
    If the ROI exceeds the defined threshold, then the agent will decide to onboard the maximum available power, 
    otherwise they will not.

    TODO:
     [ ] - 
    """
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, max_daily_rb_onboard_pib=3,
                 renewal_rate = 0.6, fil_plus_rate=0.6, 
                 agent_optimism=4, roi_threshold=0.1):
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)

        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate

        self.roi_threshold = roi_threshold
        self.agent_optimism = agent_optimism

        self.duration_vec_days = (np.asarray([12, 36, 60])*30).astype(np.int32)  # 1Y, 3Y, 5Y sectors are possible

        self.map_optimism_scales()

        for d in self.duration_vec_days:
            self.agent_info_df[f'roi_estimate_{d}'] = 0

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


    def estimate_roi(self, sector_duration, date_in):
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]

        # NOTE: we need to use yesterday's metrics b/c today's haven't yet been aggregated by the system yet
        prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']

        # TODO: make this an iterative update rather than full-estimate every day
        # NOTE: this assumes that the pledge remains constant. This is not true, but a zeroth-order approximation
        future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        
        # get the cost per sector for the duration, which in this case is just borrowing costs
        sector_duration_yrs = sector_duration / 360.0
        pledge_repayment_estimate = self.compute_repayment_amount_from_supply_discount_rate_model(date_in, prev_day_pledge_per_QAP, sector_duration_yrs)
        cost_per_sector_estimate = pledge_repayment_estimate - prev_day_pledge_per_QAP
        roi_estimate = (future_rewards_per_sector_estimate.sum() - cost_per_sector_estimate) / prev_day_pledge_per_QAP
        
        # annualize it so that we can have the same frame of reference when comparing different sector durations
        duration_yr = sector_duration / 360.0  
        roi_estimate_annualized = (1.0+roi_estimate)**(1.0/duration_yr) - 1
        
        # if np.isnan(future_rewards_per_sector_estimate.sum()) or np.isnan(prev_day_pledge_per_QAP) or np.isnan(roi_estimate) or np.isnan(roi_estimate_annualized):
        #     print(self.unique_id, future_rewards_per_sector_estimate.sum(), prev_day_pledge_per_QAP, roi_estimate, roi_estimate_annualized)

        return roi_estimate_annualized

    def step(self):
        roi_estimate_vec = []
        
        agent_df_idx = self.agent_info_df[pd.to_datetime(self.agent_info_df['date']) == pd.to_datetime(self.current_date)].index[0]
        for d in self.duration_vec_days:    
            roi_estimate = self.estimate_roi(d, self.current_date)            
            roi_estimate_vec.append(roi_estimate)
            self.agent_info_df.loc[agent_df_idx, 'roi_estimate_%d' % (d,)] = roi_estimate
            
        max_roi_idx = np.argmax(roi_estimate_vec)
        best_duration = self.duration_vec_days[max_roi_idx]
        if roi_estimate_vec[max_roi_idx] > self.roi_threshold:
            # for now, we put all power into FIL+ (deal power)
            rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
            qa_to_onboard = apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate)
            pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)

            total_qa_onboarded = rb_to_onboard + qa_to_onboard
            pledge_needed_for_onboarding = total_qa_onboarded * pledge_per_pib
            pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                        pledge_needed_for_onboarding, 
                                                                                                        best_duration)

            self.onboard_power(self.current_date, rb_to_onboard, total_qa_onboarded, best_duration,
                               pledge_needed_for_onboarding, pledge_repayment_value_onboard)

            # renew available power for the same duration
            se_power_dict = self.get_se_power_at_date(self.current_date)
            # only renew CC power
            cc_power_to_renew = se_power_dict['se_cc_power'] * self.renewal_rate
            pledge_needed_for_renewal = cc_power_to_renew * pledge_per_pib
            pledge_repayment_value_renew = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                         pledge_needed_for_renewal, 
                                                                                                         best_duration)

            self.renew_power(self.current_date, cc_power_to_renew, best_duration,
                             pledge_needed_for_renewal, pledge_repayment_value_renew)

        super().step()

    def post_global_step(self):
        # we can update local representation of anything else that should happen after
        # global metrics for day are aggregated
        pass