from datetime import datetime, timedelta, date
from .. import constants
from .sp_agent import SPAgent
from ..power import cc_power, deal_power


import numpy as np
import pandas as pd

def linear(x1, y1, x2, y2, x):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m * x + b

class ROIAgentDynamicOnboardandTerminateWaves(SPAgent):
    """
    The ROI agent is an agent that uses ROI forecasts to decide how much power to onboard.
    It uses a linear function to go between min/max onboard after ROI exceeds threshold

    TODO:
     [ ] - 
    """
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, 
                 min_daily_rb_onboard_pib=3, max_daily_rb_onboard_pib=12,
                 min_renewal_rate = 0.3, max_renewal_rate = 0.8,
                 fil_plus_rate=0.6, 
                 agent_optimism=4, min_roi=0.1, max_roi=0.3, renewal_rate=0.6, sector_duration = 354, terminate_date=None, future_exchange_rate = None,  termination_fee_days = 90, onboarding_coefficient = 1, agent_type = None, debug_mode=False):
        """

        debug_mode - if True, the agent will compute the power scheduled to be onboarded/renewed, but will not actually
                     onboard/renew that power, but rather return the values.  This can be used for debugging
                     or other purposes
        """
        # note that we dont set renewal_rate, roi_threshold since we use those terms differently in this agent
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)

        self.sector_duration = sector_duration
        self.sector_duration_yrs = sector_duration / 365
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.terminate_date = terminate_date
        self.future_exchange_rate = future_exchange_rate
        self.onboarding_coefficient = onboarding_coefficient
        self.termination_fee_days = termination_fee_days
        self.agent_type = agent_type


        self.min_daily_rb_onboard_pib = min_daily_rb_onboard_pib
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.min_renewal_rate = min_renewal_rate
        self.max_renewal_rate = max_renewal_rate

        self.min_roi = min_roi
        self.max_roi = max_roi

        self.fil_plus_rate = fil_plus_rate

        self.agent_optimism = agent_optimism

        self.duration_vec_days = np.asarray([360, 360*3]).astype(np.int32)  # 1Y, 3Y, 5Y sectors are possible

        self.map_optimism_scales()

        for d in self.duration_vec_days:
            self.agent_info_df[f'roi_estimate_{d}'] = 0

        self.debug_mode = debug_mode

    ##########################################################################################
    # TODO: better to use the superclass construct, but we need to figure out 
    # how to get teh grandparent b/c we need to step on grandparent, not parent.
    # Copilot says this, but it doesnt seem to work:
    """
    class Grandparent:
        pass
    class Parent(Grandparent):
        pass
    class Child(Parent):
        pass
    child = Child()
    grandparent = super(Child, child).__class__.__bases__[0]
    """
    ##########################################################################################
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

        # print(date_in, 'prev_day_pledge_per_QAP', prev_day_pledge_per_QAP)

        # TODO: make this an iterative update rather than full-estimate every day
        # NOTE: this assumes that the pledge remains constant. This is not true, but a zeroth-order approximation
        future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        
        # get the cost per sector for the duration, which in this case is just borrowing costs
        sector_duration_yrs = sector_duration / 360.
        pledge_repayment_estimate = self.compute_repayment_amount_from_supply_discount_rate_model(date_in, prev_day_pledge_per_QAP, sector_duration_yrs)
        cost_per_sector_estimate = pledge_repayment_estimate - prev_day_pledge_per_QAP
        if prev_day_pledge_per_QAP == 0:
            roi_estimate = self.max_roi
        else:
            roi_estimate = (future_rewards_per_sector_estimate.sum() - cost_per_sector_estimate) / prev_day_pledge_per_QAP
        
        # annualize it so that we can have the same frame of reference when comparing different sector durations
        if roi_estimate < -1:
            roi_estimate_annualized = self.roi_threshold - 1  # if ROI is too low, set it so that it doesn't onboard.
                                                              # otherwise, you would take an exponent of a negative number
                                                              # to a fractional power below and get a complex number
        else:
            roi_estimate_annualized = (1.0+roi_estimate)**(1.0/sector_duration_yrs) - 1
        
        # print(roi_estimate, roi_estimate_annualized, duration_yr)
        # if np.isnan(future_rewards_per_sector_estimate.sum()) or np.isnan(prev_day_pledge_per_QAP) or np.isnan(roi_estimate) or np.isnan(roi_estimate_annualized):
        #     print(self.unique_id, future_rewards_per_sector_estimate.sum(), prev_day_pledge_per_QAP, roi_estimate, roi_estimate_annualized)

        return roi_estimate_annualized
    ##########################################################################################

    def get_returns(self, sector_duration, date_in):
        filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]
        prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']
        future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        return(future_rewards_per_sector_estimate).sum()

    def get_termination_fee(self, date_in):
      terminate_date = self.current_date
      rb_active_power, qa_active_power, termination_fee, associated_pledge_to_release, date_to_release = self._trace_modeled_power(self.start_date, terminate_date)
      num_sectors_to_terminate = qa_active_power * constants.PIB / constants.SECTOR_SIZE
      if num_sectors_to_terminate != 0:
        termination_fee = termination_fee/num_sectors_to_terminate
      else:
        termination_fee = 0

      termination_fee = (self.termination_fee_days/90)*termination_fee

      return rb_active_power, qa_active_power, termination_fee

    def estimate_utility_from_termination(self, date_in):     
      terminate_date = self.current_date
      # rb_active_power, qa_active_power, termination_fee, associated_pledge_to_release, date_to_release = self._trace_modeled_power(self.start_date, terminate_date)
      rb_active_power, qa_active_power, termination_fee = self.get_termination_fee(date_in)
      termination_fee_fiat = 4.5*termination_fee

      # print('termination fee {}'.format(str(termination_fee)))
      # print('termination fee fiat {}'.format(str(termination_fee_fiat)))

      salvage_value = 4.5*((1/4)*(0.4))
      utility_estimate = salvage_value - termination_fee_fiat

      return (utility_estimate)

    def estimate_utility_from_no_termination(self, sector_duration, date_in):
      filecoin_df_idx = self.model.filecoin_df[pd.to_datetime(self.model.filecoin_df['date']) == pd.to_datetime(date_in)].index[0]

        # NOTE: we need to use yesterday's metrics b/c today's haven't yet been aggregated by the system yet
      prev_day_pledge_per_QAP = self.model.filecoin_df.loc[filecoin_df_idx-1, 'day_pledge_per_QAP']

        # print(date_in, 'prev_day_pledge_per_QAP', prev_day_pledge_per_QAP)

        # TODO: make this an iterative update rather than full-estimate every day
        # NOTE: this assumes that the pledge remains constant. This is not true, but a zeroth-order approximation
      future_rewards_per_sector_estimate = self.forecast_day_rewards_per_sector(date_in, sector_duration)
        
        # get the cost per sector for the duration, which in this case is just borrowing costs
      sector_duration_yrs = sector_duration / 360.
      # pledge_repayment_estimate = self.compute_repayment_amount_from_supply_discount_rate_model(date_in, prev_day_pledge_per_QAP, sector_duration_yrs)
      # cost_per_sector_estimate = pledge_repayment_estimate - prev_day_pledge_per_QAP
      
      utility_estimate = future_rewards_per_sector_estimate.sum() 
      utility_estimate = self.future_exchange_rate*utility_estimate

      return utility_estimate

    def convert_roi_to_onboard(self, estimated_roi):
        """
        Returns the amount of power to onboard and % to renew based on the delta between
        the ROI threshold and the estimated ROI
        """
        rb_to_onboard = linear(self.min_roi, self.min_daily_rb_onboard_pib, self.max_roi, self.max_daily_rb_onboard_pib, estimated_roi)
        renew_pct = linear(self.min_roi, self.min_renewal_rate, self.max_roi, self.max_renewal_rate, estimated_roi)
        return rb_to_onboard, renew_pct

    def step(self):
        # utility_compute_duration_yrs =  1
        # utility_compute_duration = 360*utility_compute_duration_yrs

        # utility_from_termination = self.estimate_utility_from_termination(self.current_date)
        # utility_from_no_termination = self.estimate_utility_from_no_termination(utility_compute_duration, self.current_date) 

        # returns = self.get_returns(utility_compute_duration, self.current_date)
        # rb_active_power, qa_active_power, termination_fee = self.get_termination_fee(self.current_date)
        passive_termination_dates = [date(2023, 8, 30), date(2024, 3, 30), date(2024, 10, 29), date(2024, 10, 30)]
        if self.agent_type is not None and self.current_date == passive_termination_dates[self.agent_type]:
          rb_active_power, qa_active_power, termination_fee = self.get_termination_fee(self.current_date)
          if (rb_active_power != 0) and (qa_active_power != 0):
            # self.terminate_all_modeled_power(self.current_date)
            self.terminate_all_known_power(self.current_date)
          else:
            pass
        elif self.agent_type is not None and self.current_date > passive_termination_dates[self.agent_type]:
          pass
        elif self.agent_type is not None and self.current_date < passive_termination_dates[self.agent_type]:
          utility_compute_duration_yrs =  1
          utility_compute_duration = 360*utility_compute_duration_yrs
          returns = self.get_returns(utility_compute_duration, self.current_date)
          rb_active_power, qa_active_power, termination_fee = self.get_termination_fee(self.current_date)

          if ((termination_fee/(returns*self.onboarding_coefficient)) < 1) and (self.terminate_date == None):
            # print('Onboarding Power')
            rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
            qa_to_onboard = self.model.apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate,
                                                       fil_plus_multipler=constants.FIL_PLUS_MULTIPLER,
                                                       date_in=self.current_date,
                                                       sector_duration_days=self.sector_duration) + \
                        rb_to_onboard * (1-self.fil_plus_rate)
            pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)

            pledge_needed_for_onboarding = qa_to_onboard * pledge_per_pib
            pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                       pledge_needed_for_onboarding, 
                                                                                                       self.sector_duration_yrs)
            if not self.debug_mode:
              self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration, 
                               pledge_needed_for_onboarding, pledge_repayment_value_onboard)
            else:
              onboard_args_to_return = (self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration, 
                                      pledge_needed_for_onboarding, pledge_repayment_value_onboard)

            if self.renewal_rate > 0:
              se_power_dict = self.get_se_power_at_date(self.current_date)
            # which aspects of power get renewed is dependent on the setting "renewals_setting" in the FilecoinModel object
              cc_power = se_power_dict['se_cc_power']
              deal_power = se_power_dict['se_deal_power']
              cc_power_to_renew = cc_power*self.renewal_rate 
              deal_power_to_renew = deal_power*self.renewal_rate  

              pledge_needed_for_renewal = (cc_power_to_renew + deal_power_to_renew) * pledge_per_pib
              pledge_repayment_value_renew = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                         pledge_needed_for_renewal, 
                                                                                                         self.sector_duration_yrs)

              if not self.debug_mode:
                  self.renew_power(self.current_date, cc_power_to_renew, deal_power_to_renew, self.sector_duration,
                                pledge_needed_for_renewal, pledge_repayment_value_renew)
              else:
                  renew_args_to_return = (self.current_date, cc_power_to_renew, deal_power_to_renew, self.sector_duration,
                                        pledge_needed_for_renewal, pledge_repayment_value_renew)

        elif self.agent_type == None and self.terminate_date is not None and self.current_date > self.terminate_date:
          pass
        elif self.agent_type == None and self.terminate_date == None:
          utility_compute_duration_yrs =  1
          utility_compute_duration = 360*utility_compute_duration_yrs

          utility_from_termination = self.estimate_utility_from_termination(self.current_date)
          utility_from_no_termination = self.estimate_utility_from_no_termination(utility_compute_duration, self.current_date) 

          returns = self.get_returns(utility_compute_duration, self.current_date)
          rb_active_power, qa_active_power, termination_fee = self.get_termination_fee(self.current_date)
          if (utility_from_termination > utility_from_no_termination) and (self.current_date >= (self.start_date + timedelta(days=180))):
            self.terminate_date = self.current_date
            # print('U1 {}'.format(str(utility_from_termination)))
            # print('U2 {}'.format(str(utility_from_no_termination)))
            self.terminate_all_modeled_power(self.current_date)
            self.terminate_all_known_power(self.current_date)
      
          if ((termination_fee/(returns*self.onboarding_coefficient)) < 1) and (self.terminate_date == None):
            # print('Onboarding Power')
            rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
            qa_to_onboard = self.model.apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate,
                                                       fil_plus_multipler=constants.FIL_PLUS_MULTIPLER,
                                                       date_in=self.current_date,
                                                       sector_duration_days=self.sector_duration) + \
                        rb_to_onboard * (1-self.fil_plus_rate)
            pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)

            pledge_needed_for_onboarding = qa_to_onboard * pledge_per_pib
            pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                       pledge_needed_for_onboarding, 
                                                                                                       self.sector_duration_yrs)
            if not self.debug_mode:
              self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration, 
                               pledge_needed_for_onboarding, pledge_repayment_value_onboard)
            else:
              onboard_args_to_return = (self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration, 
                                      pledge_needed_for_onboarding, pledge_repayment_value_onboard)

            if self.renewal_rate > 0:
              se_power_dict = self.get_se_power_at_date(self.current_date)
            # which aspects of power get renewed is dependent on the setting "renewals_setting" in the FilecoinModel object
              cc_power = se_power_dict['se_cc_power']
              deal_power = se_power_dict['se_deal_power']
              cc_power_to_renew = cc_power*self.renewal_rate 
              deal_power_to_renew = deal_power*self.renewal_rate  

              pledge_needed_for_renewal = (cc_power_to_renew + deal_power_to_renew) * pledge_per_pib
              pledge_repayment_value_renew = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                         pledge_needed_for_renewal, 
                                                                                                         self.sector_duration_yrs)

              if not self.debug_mode:
                  self.renew_power(self.current_date, cc_power_to_renew, deal_power_to_renew, self.sector_duration,
                                pledge_needed_for_renewal, pledge_repayment_value_renew)
              else:
                  renew_args_to_return = (self.current_date, cc_power_to_renew, deal_power_to_renew, self.sector_duration,
                                        pledge_needed_for_renewal, pledge_repayment_value_renew)

        # even if we are in debug mode, we need to step the agent b/c that updates agent internal states
        # such as current_date
        super().step()

        if self.debug_mode:
            return onboard_args_to_return, renew_args_to_return
