from datetime import timedelta
from . import constants
from .sp_agent import SPAgent
from .power import cc_power, deal_power

class SPAgent_Backtest(SPAgent):
    """
    For backtesting, end_date should NOT EXCEED the date past which data for 
    the network is unknown.  This is because the agent will be seeded with
    historical power information to ensure that the remainder of the simulation
    produces expected results for circulating supply, etc.
    """
    def __init__(self, model, id, historical_power, start_date, end_date):
        super().__init__(model, id, historical_power, start_date, end_date)

    def seed_backtest(self, power_df):
        """
        power_df - the power information from start_date --> end_date

        TODO: you can also try modeling the SE power with a static duration
        rather than seeding it exactly. This may have some complications b/c
        the known SE power doesn't have an attached duration.
        """
        # the agent was seeded with historical power already, which goes from
        # network_start --> start_date.  Since this is a backtesting agent, we
        # want to distribute the remainder of the power from start_date --> end_date
        # and then run the agent as if it is making decisions to onboard/renew/terminate
        # the amount of power that was configured

        # simulations always start from the date: NETWORK_DATA_START
        global_ii = (power_df.iloc[0]['date'] - constants.NETWORK_DATA_START).days
        ii_start = global_ii
        for _, row in power_df.iterrows():
            self.onboarded_power[global_ii] = (
                cc_power(row['day_onboarded_rb_power_pib']),
                deal_power(row['day_onboarded_qa_power_pib'])
            )
            self.renewed_power[global_ii] = (
                cc_power(row['extended_rb']),
                deal_power(row['extended_qa'])
            )
            self.scheduled_expire_power[global_ii] = (
                cc_power(row['total_rb']),
                deal_power(row['total_qa'])
            )
            self.terminated_power[global_ii] = (
                cc_power(row['terminated_rb']),
                deal_power(row['terminated_qa'])
            )
            global_ii += 1

        # print("Backtest Seeding agent: %d from index=%d:%d" % (self.unique_id, ii_start, global_ii))

    def step(self):
        # nothing needs to happen in the step function b/c the 
        # vectors were already seeded with the bactesting information, so 
        # effectively the backtesting data already made all of the decisions.

        # call the book-keeping stuff.  We do this manually instead of
        # calling the super() because we don't want to auto-update the SE power
        self.current_day += 1
        self.current_date += timedelta(days=1)
