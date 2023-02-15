import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin


def power_proportional_capital_distribution_policy(miner_qa_power=0, total_qa_power=1, total_capital_inflow=1):
    """
    This policy distributes capital to miners proportional to their power.
    """
    return min(miner_qa_power/total_qa_power, 1.0) * total_capital_inflow


class PID:
    def __init__(self, p=0.1, i=0.1, d=0.1, setpoint=0.5,
                 clip_low=0, clip_high=1):
        self.p = p
        self.i = i
        self.d = d
        self.clip_low = clip_low
        self.clip_high = clip_high
        
        self.setpoint = setpoint
        
        self.err_sum = 0 
        self.last_err = 0
        
    def change_setpoint(self, p):
        self.setpoint = p
        
    def step(self, v):
        # v is the current value
        # setpoint is our target
        
        err = self.setpoint - v  # if v is less, err is +, if v is more, err is -
        self.err_sum += err
        d_err = err - self.last_err
        
        output = v + (err * self.p) + (self.i * self.err_sum) + (self.d * d_err)
        output = np.clip(output, self.clip_low, self.clip_high)
        self.last_err = err
        return output


class MultiplyXform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['mkt_cap_crude'] = X['circ_supply'] * X['price_Q50']
        
        # enable single sample predictions
        out = X[['mkt_cap_crude']].values
        if out.shape[0] == 1:
            out = out.reshape(1, -1)
        
        return out


class CapitalInflowProcess:
    """
    This class models capital inflow into the Filecoin network.

    It works as follows:
     - We use a linear model to predict the market-cap, using the following features:
        - circ-supply
        - price
     - User defines set-points for the percentage of circulating supply that will be
       infilled into the Filecoin market, given the sign of the gradient of the market-cap
       and the sign of the gradient of the network QAP.
     - A PID controller then tracks the set-points

    TODO
    [ ] - It would be good to try to predict market-cap by only using circ-supply, rather
          than also using price because price is forecasted by the price_process, so there is
          a sort of circular dependency.
          @tmellan - thoughts?
    [ ] - measure accuracy by sign of gradient, rather than predicting actual market-cap
          the downside of this is that it is specific to the current model which uses
          signs of qap & market-cap w/ a PID controller ..
    """
    def __init__(self, filecoin_model, pid_config=None, debug_model=False):
        self.model = filecoin_model
        self.debug_model = debug_model

        if pid_config is None:
            p = 0.1
            i = 0.01
            d = 0.01
            
            # the set-points are the target % of circulating supply that will be infilled
            # into the Filecoin network, given the sign of the gradient of the market-cap
            setpoint_pos_pos = .01
            setpoint_pos_neg = 0.005
            setpoint_neg_pos = 0.0025
            setpoint_neg_neg = 0
        self.setpoint_pos_pos = setpoint_pos_pos
        self.setpoint_pos_neg = setpoint_pos_neg
        self.setpoint_neg_pos = setpoint_neg_pos
        self.setpoint_neg_neg = setpoint_neg_neg

        # PID values were found with trial and error
        clip_low = min(setpoint_neg_neg, setpoint_neg_pos, setpoint_pos_neg, setpoint_pos_pos)
        clip_high = max(setpoint_neg_neg, setpoint_neg_pos, setpoint_pos_neg, setpoint_pos_pos)
        self.pid_controller = PID(p, i, d, clip_low=clip_low, clip_high=clip_high)
        
        self.prepare_train_data()
        self.train_mkt_cap_model()
    
    def prepare_train_data(self):
        sim_start_idx = self.model.filecoin_df[self.model.filecoin_df['date'] == self.model.start_date].index[0]
        network_historical_data = self.model.filecoin_df.iloc[:sim_start_idx]

        #self.train_data = network_historical_data[['date', 'circ_supply', 'network_locked', 'total_qa_power_eib', 'day_pledge_per_QAP', 'day_rewards_per_sector']]
        self.train_data = network_historical_data[['date', 'circ_supply']]
        market_data = self.model.global_forecast_df[['date', 'market_caps', 'price_Q50']]
        self.train_data = self.train_data.merge(market_data, on='date', how='inner').dropna()

    def train_mkt_cap_model(self):
        data = self.train_data[['date', 'circ_supply', 'price_Q50', 'market_caps']]
        # print(len(data), data.iloc[0]['date'], data.iloc[-1]['date'])
        X = data[['circ_supply', 'price_Q50']]
        y = data['market_caps']

        self.market_cap_prediction_model = make_pipeline(MultiplyXform(), 
                                                         StandardScaler(), 
                                                         SGDRegressor(max_iter=1000, tol=1e-3))

        if self.debug_model:
            import pickle
            with open('data_in.pkl', 'wb') as f:
                data_in = {
                    'X': X,
                    'y': y
                }
                pickle.dump(data_in, f)
            tscv = TimeSeriesSplit()
            cv_results = cross_validate(self.market_cap_prediction_model, X, y, cv=tscv, scoring='r2')
            print('CV Scores - Model Fit', cv_results['test_score'], np.mean(cv_results['test_score']))

        self.market_cap_prediction_model.fit(X, y)  # train on full data for usage

    def determine_setpoint(self, mkt_cap_grad, qap_grad):
        if mkt_cap_grad > 0 and qap_grad > 0:
            return self.setpoint_pos_pos
        elif mkt_cap_grad > 0 and qap_grad < 0:
            return self.setpoint_pos_neg
        elif mkt_cap_grad < 0 and qap_grad > 0:
            return self.setpoint_neg_pos
        elif mkt_cap_grad < 0 and qap_grad < 0:
            return self.setpoint_neg_neg
        else:
            return None

    def step(self):
        # get the data to predict mkt-cap
        filecoin_df_idx = self.model.filecoin_df[self.model.filecoin_df['date'] == self.model.current_date].index[0]
        cs = self.model.filecoin_df.loc[filecoin_df_idx, 'circ_supply']
        global_forecast_df_idx = self.model.global_forecast_df[self.model.global_forecast_df['date'] == self.model.current_date].index[0]
        price = self.model.global_forecast_df.loc[global_forecast_df_idx]['price_Q50']

        # arrange in correct way
        X_in = pd.DataFrame({'circ_supply': [cs], 'price_Q50': [price]})

        # make prediction
        mkt_cap_pred = self.market_cap_prediction_model.predict(X_in)

        # write it into the global forecast
        self.model.global_forecast_df.loc[global_forecast_df_idx, 'market_caps'] = mkt_cap_pred

        # update the infil percentage
        mkt_cap_grad = self.model.global_forecast_df.loc[global_forecast_df_idx, 'market_caps'] - self.model.global_forecast_df.loc[global_forecast_df_idx-1, 'market_caps']
        qap_grad = self.model.filecoin_df.loc[filecoin_df_idx, 'total_qa_power_eib'] - self.model.filecoin_df.loc[filecoin_df_idx-1, 'total_qa_power_eib']

        setpoint = self.determine_setpoint(mkt_cap_grad, qap_grad)
        if setpoint is not None:
            self.pid_controller.change_setpoint(setpoint)
        
        infil_pct = self.pid_controller.step(self.model.filecoin_df.loc[filecoin_df_idx-1, 'capital_inflow_pct'])

        self.model.filecoin_df.loc[filecoin_df_idx, 'capital_inflow_pct'] = infil_pct
        self.model.filecoin_df.loc[filecoin_df_idx, 'capital_inflow_FIL'] = infil_pct / 100. * cs


class CapitalInflowProcess_OLD:
    """
    This class models capital inflow into the Filecoin network.
    It works as follows:



    Idea for this comes from Tom Mellan.
    """

    def __init__(self, filecoin_model, conv_filter=None, debug_model=False):
        """
        NOTE: this object should be initiailized after the model is run and seeded
        with initial data regarding network power, roi, and circ-supply related quantities.

        """
        self.model = filecoin_model
        if conv_filter is None:
            self.conv_filter = np.ones(7)/7.  # 7 day moving average
        else:
            self.conv_filter = conv_filter
        self.window_len = len(self.conv_filter)
        self.debug_model = debug_model

        # self.prepare_train_data()
        # self.train_mkt_cap_model()

    # def prepare_train_data(self):
    #     sim_start_idx = self.model.filecoin_df[self.model.filecoin_df['date'] == self.model.start_date].index[0]
    #     network_historical_data = self.model.filecoin_df.iloc[:sim_start_idx]

    #     self.train_data = network_historical_data[['date', 'circ_supply', 'network_locked', 'total_qa_power_eib', 'day_pledge_per_QAP', 'day_rewards_per_sector']]
    #     market_data = self.model.global_forecast_df[['date', 'market_caps', 'price_Q50']]
    #     self.train_data = self.train_data.merge(market_data, on='date', how='left')

    # def train_mkt_cap_model(self):
    #     data = self.train_data[['circ_supply', 'network_locked', 'total_qa_power_eib', 'day_pledge_per_QAP', 'day_rewards_per_sector', 'price_Q50', 'market_caps']]
        
    #     data['qap_grad'] = data['total_qa_power_eib'].diff()
    #     data['locked_div_supply'] = data['network_locked'] / data['circ_supply']
        
    #     data = data.dropna()

    #     X = data[['locked_div_supply', 'total_qa_power_eib', 'qap_grad', 'day_pledge_per_QAP', 'day_rewards_per_sector', 'price_Q50']]
    #     y = data['market_caps']

    #     if self.debug_model:
    #         clf = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    #         cv_scores = cross_val_score(clf, X.values, y.values, cv=5)
    #         print('CV Scores - Model Fit', cv_scores)

    #     self.market_cap_prediction_model = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
    #     self.market_cap_prediction_model.fit(X.values, y.values)  # train on full data for usage

    # def step_old(self):
    #     # get predictor variables for market-cap for current day
    #     filecoin_df_idx = self.model.filecoin_df[self.model.filecoin_df['date'] == self.model.current_date].index[0]
    #     day_data = self.model.filecoin_df.iloc[filecoin_df_idx][['circ_supply', 'network_locked', 'total_qa_power_eib', 'day_pledge_per_QAP', 'day_rewards_per_sector']]
    #     day_data['locked_div_supply'] = day_data['network_locked'] / day_data['circ_supply']
    #     day_data['qap_grad'] = day_data['total_qa_power_eib'] - self.model.filecoin_df.iloc[filecoin_df_idx-1]['total_qa_power_eib']
        
    #     global_forecast_df_idx = self.model.global_forecast_df[self.model.global_forecast_df['date'] == self.model.current_date].index[0]
    #     day_data['price_Q50'] = self.model.global_forecast_df.loc[global_forecast_df_idx]['price_Q50']

    #     X = day_data[['locked_div_supply', 'total_qa_power_eib', 'qap_grad', 'day_pledge_per_QAP', 'day_rewards_per_sector', 'price_Q50']].values.reshape(1, -1)
    #     # print(X)
    #     mkt_cap_pred = self.market_cap_prediction_model.predict(X)
    #     self.update_global_forecast_df(filecoin_df_idx, global_forecast_df_idx, mkt_cap_pred)

    def step(self):
        filecoin_df_idx = self.model.filecoin_df[self.model.filecoin_df['date'] == self.model.current_date].index[0]
        cs = self.model.filecoin_df.loc[filecoin_df_idx, 'circ_supply']
        global_forecast_df_idx = self.model.global_forecast_df[self.model.global_forecast_df['date'] == self.model.current_date].index[0]
        price = self.model.global_forecast_df.loc[global_forecast_df_idx]['price_Q50']
        mkt_cap_pred = cs * price
        self.update_global_forecast_df(filecoin_df_idx, global_forecast_df_idx, mkt_cap_pred)
    

    def update_global_forecast_df(self, filecoin_df_idx, global_forecast_df_idx, mkt_cap_pred):
        # write this into the global forecast df
        self.model.global_forecast_df.loc[global_forecast_df_idx, 'market_caps'] = mkt_cap_pred

        # get market-cap growth rate and smooth it with the convolutional filter
        # print(self.model.current_date, global_forecast_df_idx)
        # print(self.model.global_forecast_df.iloc[global_forecast_df_idx]['date'], self.model.global_forecast_df.iloc[global_forecast_df_idx-self.window_len]['date'])
        mkt_cap_diff = self.model.global_forecast_df.loc[global_forecast_df_idx-self.window_len:global_forecast_df_idx, 'market_caps'].diff().dropna().values
        mkt_cap_prev = self.model.global_forecast_df.loc[global_forecast_df_idx-self.window_len-1:global_forecast_df_idx-2, 'market_caps'].values
        
        mkt_cap_growth_rate = mkt_cap_diff / mkt_cap_prev
        mkt_cap_growth_rate_smoothed = np.convolve(mkt_cap_growth_rate, self.conv_filter, mode='valid')[0]
        
        # TODO: Revisit
        # determine capital inflow
        # means that the inflow is 0-1% of circ-supply on any given day, depending on market cap. revisit as this seems low
        # this is a value between 0 and 1, we just use it as a percentage

        # see here for sigmoid scaling: https://math.stackexchange.com/questions/1214167/
        # remap the sigmoid function such that a growth-rate of -1 maps to 0% infil and 1 maps to 1 inflow
        # scale = 1 * np.log(2 + np.sqrt(3))
        # capital_inflow_pct = expit(scale*mkt_cap_growth_rate_smoothed)  # sigmoid function
        capital_inflow_pct = np.clip(mkt_cap_growth_rate_smoothed, 0, 1)
        self.model.filecoin_df.loc[filecoin_df_idx, 'capital_inflow_pct'] = capital_inflow_pct
        
        self.model.filecoin_df.loc[filecoin_df_idx, 'capital_inflow_FIL'] = capital_inflow_pct / 100. * self.model.filecoin_df.loc[filecoin_df_idx, 'circ_supply']
        
