from datetime import timedelta
from .. import constants
from .sp_agent import SPAgent
from ..power import cc_power, deal_power

import numpy as np
import pandas as pd

class DCAAgent(SPAgent):
    """
    The Dollar-Cost-Averaging agent is a simple agent that onboards a fixed amount of power every day
    This is based on the common investment strategy of dollar-cost-averaging.

    TODO
     [ ] - vectorize the onboarding, renewal rate and FIL+ rates
    """
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, max_daily_rb_onboard_pib=3, 
                 renewal_rate=0.6, fil_plus_rate=0.6, sector_duration=365, debug_mode=False):
        """

        debug_mode - if True, the agent will compute the power scheduled to be onboarded/renewed, but will not actually
                     onboard/renew that power, but rather return the values.  This can be used for debugging
                     or other purposes
        """
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)
        
        self.sector_duration = sector_duration
        self.sector_duration_yrs = sector_duration / 365
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.debug_mode = debug_mode

    def step(self):
        rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
        qa_to_onboard = self.model.apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate,
                                                       fil_plus_multipler=constants.FIL_PLUS_MULTIPLER,
                                                       date_in=self.current_date,
                                                       sector_duration_days=self.sector_duration) + \
                        rb_to_onboard * (1-self.fil_plus_rate)
        pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)
        
        # debugging to ensure that pledge/sector seems reasonable
        # sector_size_in_pib = constants.SECTOR_SIZE / constants.PIB
        # pledge_per_sector = self.model.estimate_pledge_for_qa_power(self.current_date, sector_size_in_pib)
        # print(pledge_per_pib, pledge_per_sector)
        
        # total_qa_onboarded = rb_to_onboard + qa_to_onboard
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

        # renewals
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

class DCAAgentLeaveNetwork(SPAgent):
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, max_daily_rb_onboard_pib=3, 
                 renewal_rate=0.6, fil_plus_rate=0.6, sector_duration=365, debug_mode=False, terminate_date=None):
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)
        
        self.sector_duration = sector_duration
        self.sector_duration_yrs = sector_duration / 365
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.debug_mode = debug_mode
        
        self.terminate_date = terminate_date

    def step(self):
        if self.terminate_date is not None and self.current_date >= self.terminate_date:
            # we stop onboarding and renewing power after this date
            super().step()
        else:
            rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
            qa_to_onboard = self.model.apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate,
                                                        fil_plus_multipler=constants.FIL_PLUS_MULTIPLER,
                                                        date_in=self.current_date,
                                                        sector_duration_days=self.sector_duration) + \
                            rb_to_onboard * (1-self.fil_plus_rate)
            pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)
            
            # debugging to ensure that pledge/sector seems reasonable
            # sector_size_in_pib = constants.SECTOR_SIZE / constants.PIB
            # pledge_per_sector = self.model.estimate_pledge_for_qa_power(self.current_date, sector_size_in_pib)
            # print(pledge_per_pib, pledge_per_sector)
            
            # total_qa_onboarded = rb_to_onboard + qa_to_onboard
            pledge_needed_for_onboarding = qa_to_onboard * pledge_per_pib
            pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                        pledge_needed_for_onboarding, 
                                                                                                        self.sector_duration_yrs)
            # print(rb_to_onboard, self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib, self.fil_plus_rate, qa_to_onboard, pledge_per_pib, pledge_needed_for_onboarding, pledge_repayment_value_onboard)
            
            if not self.debug_mode:
                self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration, 
                                pledge_needed_for_onboarding, pledge_repayment_value_onboard)
            else:
                onboard_args_to_return = (self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration, 
                                        pledge_needed_for_onboarding, pledge_repayment_value_onboard)

            # renewals
            if self.renewal_rate > 0:
                se_power_dict = self.get_se_power_at_date(self.current_date)
                # which aspects of power get renewed is dependent on the setting "renewals_setting" in the FilecoinModel object
                cc_power = se_power_dict['se_cc_power']
                deal_power = se_power_dict['se_deal_power']
                cc_power_to_renew = cc_power*self.renewal_rate 
                deal_power_to_renew = deal_power*self.renewal_rate  

                # print('DCATerminate[%d]:' % (self.unique_id,), se_power_dict, cc_power_to_renew, deal_power_to_renew)

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
        
class DCAAgentTerminate(SPAgent):
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, max_daily_rb_onboard_pib=3, 
                 renewal_rate=0.6, fil_plus_rate=0.6, sector_duration=365, debug_mode=False, terminate_date=None):
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)
        
        self.sector_duration = sector_duration
        self.sector_duration_yrs = sector_duration / 365
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate
        self.debug_mode = debug_mode
        
        self.terminate_date = terminate_date

    def step(self):
        if self.terminate_date is not None and self.current_date == self.terminate_date:
            self.terminate_all_modeled_power(self.current_date)
            self.terminate_all_known_power(self.current_date)
        elif self.terminate_date is not None and self.current_date > self.terminate_date:
            # we stop onboarding and renewing power after this date
            pass
        else:
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

            # renewals
            if self.renewal_rate > 0:
                se_power_dict = self.get_se_power_at_date(self.current_date)
                # which aspects of power get renewed is dependent on the setting "renewals_setting" in the FilecoinModel object
                cc_power = se_power_dict['se_cc_power']
                deal_power = se_power_dict['se_deal_power']
                cc_power_to_renew = cc_power*self.renewal_rate 
                deal_power_to_renew = deal_power*self.renewal_rate  

                # print('DCATerminate[%d]:' % (self.unique_id,), se_power_dict, cc_power_to_renew, deal_power_to_renew)

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