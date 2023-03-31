from datetime import timedelta
from . import constants
from .sp_agent import SPAgent
from ..power import cc_power, deal_power
from ..filecoin_model import apply_qa_multiplier

import numpy as np
import pandas as pd

class BasicRationalAgent(SPAgent):
    """
    The Dollar-Cost-Averaging agent is a simple agent that onboards a fixed amount of power every day
    given the amount of FIL it has available. Currently, this agent will only choose 1Y sectors.
    This is based on the common investment strategy of dollar-cost-averaging.

    Currently - this agent only adds FIL+ power to the network.

    TODO:
     [ ] - CC/Deal power split - this can be an input config option for the agent
     [ ] - Sector duration - this can be an input config option for the agent
    """
    def __init__(self, model, id, historical_power, start_date, end_date,
                 max_sealing_throughput=constants.DEFAULT_MAX_SEALING_THROUGHPUT_PIB, max_daily_rb_onboard_pib=3,
                 renewal_rate=0.6, fil_plus_rate=0.6, sector_duration=360,
                 discount_rate_floor_pct=20):
        super().__init__(model, id, historical_power, start_date, end_date, max_sealing_throughput_pib=max_sealing_throughput)

        self.sector_duration = sector_duration
        self.sector_duration_yrs = sector_duration / 360
        self.max_daily_rb_onboard_pib = max_daily_rb_onboard_pib
        self.renewal_rate = renewal_rate
        self.fil_plus_rate = fil_plus_rate

        self.discount_rate_floor_pct = discount_rate_floor_pct
        self.discount_rate_floor = discount_rate_floor_pct / 100.

    def scale_power_by_discount_rate(self, power_pib, current_discount_rate_pct):
        # linearly decrease the amount of power we onboard as the discount rate increases
        # TODO: other ways to modulate this onboarding can potentially yield interesting simulation results
        power_onboard_ratio = self.discount_rate_floor_pct / current_discount_rate_pct
        return power_pib * power_onboard_ratio

    def step(self):
        rb_to_onboard = min(self.max_daily_rb_onboard_pib, self.max_sealing_throughput_pib)
        qa_to_onboard = apply_qa_multiplier(rb_to_onboard * self.fil_plus_rate)
        
        current_discount_rate_pct = self.model.get_discount_rate_pct(self.current_date)
        rb_to_onboard = self.scale_power_by_discount_rate(rb_to_onboard, current_discount_rate_pct)
        qa_to_onboard = self.scale_power_by_discount_rate(qa_to_onboard, current_discount_rate_pct)
        
        pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)
        
        # debugging to ensure that pledge/sector seems reasonable
        # sector_size_in_pib = constants.SECTOR_SIZE / constants.PIB
        # pledge_per_sector = self.model.estimate_pledge_for_qa_power(self.current_date, sector_size_in_pib)
        # print(pledge_per_pib, pledge_per_sector)
        
        total_qa_onboarded = rb_to_onboard + qa_to_onboard
        pledge_needed_for_onboarding = total_qa_onboarded * pledge_per_pib
        pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                       pledge_needed_for_onboarding, 
                                                                                                       self.sector_duration_yrs)

        self.onboard_power(self.current_date, rb_to_onboard, total_qa_onboarded, self.sector_duration, 
                           pledge_needed_for_onboarding, pledge_repayment_value_onboard)

        # renewals
        if self.renewal_rate > 0:
            se_power_dict = self.get_se_power_at_date(self.current_date)
            # only renew CC power
            cc_power = se_power_dict['se_cc_power']
            cc_power_to_renew = cc_power*self.renewal_rate  # we don't cap renewals, TODO: check whether this is a reasonable assumption
            cc_power_to_renew = self.scale_power_by_discount_rate(qa_to_onboard, current_discount_rate_pct)

            pledge_needed_for_renewal = cc_power_to_renew * pledge_per_pib
            pledge_repayment_value_renew = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, 
                                                                                                         pledge_needed_for_renewal, 
                                                                                                         self.sector_duration_yrs)

            self.renew_power(self.current_date, cc_power_to_renew, self.sector_duration,
                             pledge_needed_for_renewal, pledge_repayment_value_renew)

        super().step()
