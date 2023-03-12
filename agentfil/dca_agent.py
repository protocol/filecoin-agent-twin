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
        self.max_daily_onboard_qap_pib = max_daily_onboard_qap_pib
        self.renewal_rate = renewal_rate

    """
    For renewal, we need:
    1 - figure out how much will expire that day
    2 - renew the defined portion of it, based on available FIL constraints
    3 - update the agent bookkeeping accordingly
    """

    def step(self):
        max_possible_qa_power = self.get_max_onboarding_qap_pib(self.current_date)

        # Compute how much FIL+ to onboard
        rb_to_onboard = min(max_possible_qa_power/constants.FIL_PLUS_MULTIPLER, self.model.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT)
        qa_to_onboard = apply_qa_multiplier(rb_to_onboard)

        qa_to_onboard = min(qa_to_onboard, self.max_daily_onboard_qap_pib)
        rb_to_onboard = qa_to_onboard/constants.FIL_PLUS_MULTIPLER

        onboard_success = self.onboard_power(self.current_date, rb_to_onboard, qa_to_onboard, self.sector_duration)
        if onboard_success:
            max_possible_qa_power -= qa_to_onboard

        # renew according to configured renewal rate
        if self.renewal_rate > 0:
            se_power_dict = self.get_se_power_at_date(self.current_date)
            # only renew CC power
            cc_power = se_power_dict['se_cc_power']
            cc_power_to_renew = cc_power*self.renewal_rate
            cc_power_to_renew = min(cc_power_to_renew, self.model.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT)
            
            # check how much we can renew based on available FIL
            max_possible_cc_power = max_possible_qa_power / constants.FIL_PLUS_MULTIPLER
            cc_power_to_renew = min(cc_power_to_renew, max_possible_cc_power)

            self.renew_power(self.current_date, cc_power_to_renew, self.sector_duration)

        super().step()