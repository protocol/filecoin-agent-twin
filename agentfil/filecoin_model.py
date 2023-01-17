import mesa
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from datetime import datetime, timedelta

from mechafil import data

from . import constants
from . import sp_agent


def solve_geometric(a, n):
    # see: https://math.stackexchange.com/a/2174287
    def f(r, a, n):
        # the geometric series 
        return a*(np.power(r,n)-1)/(r-1) - 1
    init_guess = 0.5
    soln = fsolve(f, init_guess, args=(a, n))
    r = soln[0]
    return r

class FilecoinModel(mesa.Model):
    def __init__(self, n, start_date, end_date):
        self.num_agents = n
        self.schedule = mesa.time.SimultaneousActivation(self)

        self.start_date = start_date
        self.current_date = start_date
        self.end_date = end_date

        self.agents = []
        self.seed_agents()

    def download_historical_data(self):
        historical_stats = data.get_historical_network_stats(
            constants.NETWORK_DATA_START,
            self.start_date,
            self.end_date
        )
        scheduled_df = data.query_sector_expirations(constants.NETWORK_DATA_START, self.end_date)
        # get the fields necessary to seed the agents into a separate dataframe that is time-aligned
        historical_stats['date'] = pd.to_datetime(historical_stats['date']).dt.date
        scheduled_df['date'] = scheduled_df['date'].dt.date
        merged_df = historical_stats.merge(scheduled_df, on='date', how='inner')
        
        # NOTE: consider using scheduled_expire_rb??
        df_historical = merged_df[
            [
                'date', 
                'day_onboarded_rb_power_pib', 'extended_rb', 'total_rb', 'terminated_rb',
                'day_onboarded_qa_power_pib', 'extended_qa', 'total_qa', 'terminated_qa',
            ]
        ]
        final_date_historical = historical_stats.iloc[-1]['date']
        df_future = scheduled_df[scheduled_df['date'] >= final_date_historical][['date', 'total_rb', 'total_qa']]

        return df_historical, df_future

    def seed_agents(self):
        df_historical, df_future = self.download_historical_data()
        
        # use a geometric-series to determine the proportion of power that goes
        # to each agent
        a = 0.2
        r = solve_geometric(a, self.num_agents)
        
        for ii in range(self.num_agents):
            agent_power_pct = a*(r**ii)
            agent_historical_df = df_historical.drop('date', axis=1) * agent_power_pct
            agent_historical_df['date'] = df_historical['date']
            agent_future_df = df_future.drop('date', axis=1) * agent_power_pct
            agent_future_df['date'] = df_future['date']
            agent_seed = {
                'historical_power': agent_historical_df,
                'future_se_power': agent_future_df
            }
            agent = sp_agent.SPAgent(ii, agent_seed, 
                                     self.start_date, self.end_date)

            self.schedule.add(agent)
            self.agents.append(
                {
                    'agent_power_pct': agent_power_pct,
                    'agent': agent
                }
            )

    def compute_macro(self, date_in):
        # compute Macro econometrics (Power statistics)
        total_rb_delta, total_qa_delta = 0, 0
        total_onboarded_rb_delta, total_renewed_rb_delta, total_se_rb_delta, total_terminated_rb_delta = 0, 0, 0, 0
        total_onboarded_qa_delta, total_renewed_qa_delta, total_se_qa_delta, total_terminated_qa_delta = 0, 0, 0, 0
        for agent_info in self.agents:
            agent = agent_info['agent']
            agent_day_power_stats = agent.get_power_at_date(date_in)
            
            total_onboarded_rb_delta += agent_day_power_stats['day_onboarded_rb_power_pib']
            total_onboarded_qa_delta += agent_day_power_stats['day_onboarded_qa_power_pib']
            
            total_renewed_rb_delta += agent_day_power_stats['extended_rb']
            total_renewed_qa_delta += agent_day_power_stats['extended_qa']

            total_se_rb_delta += agent_day_power_stats['total_rb']
            total_se_qa_delta += agent_day_power_stats['total_qa']

            total_terminated_rb_delta += agent_day_power_stats['terminated_rb']
            total_terminated_qa_delta += agent_day_power_stats['terminated_qa']

        total_rb_delta += (total_onboarded_rb_delta + total_renewed_rb_delta - total_se_rb_delta - total_terminated_rb_delta)
        total_qa_delta += (total_onboarded_qa_delta + total_renewed_qa_delta - total_se_qa_delta - total_terminated_qa_delta)

        out_dict = {
            'date': date_in,
            'day_onboarded_rb_power_pib': total_onboarded_rb_delta,
            'day_onboarded_qa_power_pib': total_onboarded_qa_delta,
            'day_extended_rb': total_renewed_rb_delta,
            'day_extended_qa': total_renewed_qa_delta,
            'day_total_rb': total_se_rb_delta,
            'day_total_qa': total_se_qa_delta,
            'day_terminated_rb': total_terminated_rb_delta,
            'day_terminated_qa': total_terminated_qa_delta,
            'day_network_rbp': total_rb_delta,
            'day_network_qap': total_qa_delta
        }
        return out_dict

    def step(self):
        # step agents
        self.schedule.step()

        macro_info = self.compute_macro(self.current_date)        

        # TODO: compute SP econometrics

        # record macro-updates that will be used for analysis

        # increment counters
        self.current_date += timedelta(days=1)