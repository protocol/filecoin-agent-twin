from datetime import timedelta
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power
from .filecoin_model import apply_qa_multiplier

import numpy as np
import pandas as pd

class DCAAgent(SPAgent):
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
                 max_daily_onboard_qap_pib=3, renewal_rate=0.6, sector_duration=360):
        super().__init__(model, id, historical_power, start_date, end_date)

        self.sector_duration = sector_duration
        self.sector_duration_yrs = sector_duration / 360
        self.max_daily_onboard_qap_pib = max_daily_onboard_qap_pib
        self.renewal_rate = renewal_rate

    def step(self):
        # TODO: agent can split this QAP into FIL+, or RB, or a combination
        # how to decide??
        # if CC, then QA = RB, if FIL+, then RB = QA / filplus_multiplier

        # for now, we put all power into FIL+ (deal power)
        rb_to_onboard = min(self.max_daily_onboard_qap_pib/constants.FIL_PLUS_MULTIPLER, self.model.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT)
        qa_to_onboard = apply_qa_multiplier(rb_to_onboard)
        pledge_per_pib = self.model.estimate_pledge_for_qa_power(self.current_date, 1.0)
        
        # debugging to ensure that pledge/sector seems reasonable
        # sector_size_in_pib = constants.SECTOR_SIZE / constants.PIB
        # pledge_per_sector = self.model.estimate_pledge_for_qa_power(self.current_date, sector_size_in_pib)
        # print(pledge_per_pib, pledge_per_sector)
        
        pledge_needed_for_onboarding = qa_to_onboard * pledge_per_pib
        pledge_repayment_value_onboard = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, pledge_needed_for_onboarding, self.sector_duration_yrs)

        # TODO: update to: put as much as possible into deal-power, and the remainder into CC power (renew first)
        self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration)

        # renewals
        if self.renewal_rate > 0:
            se_power_dict = self.get_se_power_at_date(self.current_date)
            # only renew CC power
            cc_power = se_power_dict['se_cc_power']
            cc_power_to_renew = cc_power*self.renewal_rate  # we don't cap renewals, TODO: check whether this is a reasonable assumption
            
            pledge_needed_for_renewal = cc_power_to_renew * pledge_per_pib
            pledge_repayment_value_renew = self.compute_repayment_amount_from_supply_discount_rate_model(self.current_date, pledge_needed_for_renewal, self.sector_duration_yrs)

            self.renew_power(self.current_date, cc_power_to_renew, self.sector_duration)

        # in the agent accounting, note how much was requested and how much was made available
        # so that we can remove the delta from the rewards later
        # TODO: it would be nice if you could remove this bookkeeping from the agent logic
        total_pledge_requested_value = pledge_needed_for_onboarding + pledge_needed_for_renewal
        total_pledge_repayment_value = pledge_repayment_value_onboard + pledge_repayment_value_renew
        self.account_pledge_repayment_FIL(self.current_date, self.sector_duration, total_pledge_requested_value, total_pledge_repayment_value)

        super().step()
