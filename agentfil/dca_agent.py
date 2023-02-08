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
                 max_daily_onboard_qap_pib=3, sector_duration=360):
        super().__init__(model, id, historical_power, start_date, end_date)

        self.sector_duration = sector_duration
        self.max_daily_onboard_qap_pib = max_daily_onboard_qap_pib

    def step(self):
        max_possible_qa_power = self.get_max_onboarding_qap_pib(self.current_date)

        # put all the power into FIL+
        rb_to_onboard = min(max_possible_qa_power/constants.FIL_PLUS_MULTIPLER, self.model.MAX_DAY_ONBOARD_RBP_PIB_PER_AGENT)
        qa_to_onboard = apply_qa_multiplier(rb_to_onboard)

        qa_to_onboard = min(qa_to_onboard, self.max_daily_onboard_qap_pib)
        rb_to_onboard = qa_to_onboard/constants.FIL_PLUS_MULTIPLER

        self.onboard_power(self.current_date, rb_to_onboard, 'cc', self.sector_duration)
        self.onboard_power(self.current_date, qa_to_onboard, 'deal', self.sector_duration)

        super().step()