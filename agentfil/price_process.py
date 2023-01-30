#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the FIL price module.
It downloads historical data for FIL token price, extracts info, and simulates
the token price based on a geometric Brownian motion

Do your own research. This is not to be considered as any sort of financial advice.

The pycoingecko package is required to obtain historical data

@author: juan.madrigalcianci
"""

import datetime

import numpy as np
import pandas as pd

import time
from datetime import datetime, timedelta

from pycoingecko import CoinGeckoAPI

from scenario_generator.gbm_forecast import gbm_forecast
from . import constants

class PriceProcess:
    '''
    This is the price process class. It is meant to simulate daily FIL price.
    
    The class first downloads historical price info via the method _getHistorical
    it then extract the relevant info, namely drift and volatility.
    Recall that, under the GBM, the price is given by 
    
    price(t)=price(0) x exp[ (drift-vol^2/2)*t  + vol*sqrt(t)*Z],
    
    with Z~N(0,1)
    
    Given this, we use the method _getParams to estimate drift and vol above. 
    Lastly, the main method is the update() function, which simulates one
    day worth of price movement. 
    '''
    def __init__(self, filecoin_model, 
                 dt:float=1., forecast_num_mc=1000, random_seed:int=1234):
        """
        """

        self.model = filecoin_model
        self.start_date = self.model.start_date
        self.end_date = self.model.end_date

        self.dt = dt
        self.forecast_num_mc = forecast_num_mc
        self.random_seed = random_seed

        self._get_historical()
        self._create_forecasts()
        self._update_model_global_forecasts()

    def _get_historical(self):
        cg = CoinGeckoAPI()
        id_ = 'filecoin'
        change_t = lambda x : datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d')
        ts = cg.get_coin_market_chart_range_by_id(id=id_,
                                                  vs_currency='usd',
                                                  from_timestamp=time.mktime(constants.NETWORK_DATA_START.timetuple()),
                                                  to_timestamp=time.mktime((self.start_date-timedelta(days=1)).timetuple()))

        self.usd_fil_exchange_df = pd.DataFrame(
            {
                "coin" : id_,
                "date" : list(map(change_t, np.array(ts['prices']).T[0])),
                "price" : np.array(ts['prices']).T[1],
                "market_caps" : np.array(ts['market_caps']).T[1], 
                "total_volumes" : np.array(ts['total_volumes']).T[1]
            }
        )
        self.usd_fil_exchange_df['date'] = pd.to_datetime(self.usd_fil_exchange_df['date']).dt.date

    def _create_forecasts(self):
        # create forecasts
        # update the prediction to something better
        last_price = self.usd_fil_exchange_df.iloc[-1]['price']
        remaining_len = (self.end_date - self.start_date).days + 1

        # use Geometric Brownian Motion to predict future prices, market caps, and total volumes
        # TODO: run prediction on market-cap & volume, but this information is not currently 
        # used by this agent, so I left it out currently
        x = self.usd_fil_exchange_df['price'].values
        # forecast_length = remaining_len + np.max(constants.MAX_SECTOR_DURATION_DAYS)
        forecast_length = remaining_len
        num_mc = self.forecast_num_mc
        seed_base = self.random_seed
        future_prices_vec = []
        for ii in range(num_mc):
            seed_in = seed_base + ii
            y = gbm_forecast(x, forecast_length, dt=self.dt, seed=seed_in)
            future_prices_vec.append(y)

        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        future_price_quantiles = np.quantile(np.asarray(future_prices_vec), quantiles, axis=0)
        
        self.future_price_df = pd.DataFrame(
            {
                "coin" : 'filecoin',
                "date" : pd.date_range(self.start_date, periods=forecast_length, freq='D'),
                "price_Q05" : future_price_quantiles[0],
                "price_Q25" : future_price_quantiles[1],
                "price_Q50" : future_price_quantiles[2],
                "price_Q75" : future_price_quantiles[3],
                "price_Q95" : future_price_quantiles[4],
                # currently market-cap and total-volume are not used, in the future this could change
                # for sophisticated agents
                "market_caps" : np.random.normal(last_price, 0.5, forecast_length),
                "total_volumes" : np.random.normal(last_price, 0.5, forecast_length)
            }
        )
        self.future_price_df['date'] = pd.to_datetime(self.future_price_df['date']).dt.date
        self.usd_fil_exchange_df = pd.concat([self.usd_fil_exchange_df, self.future_price_df], ignore_index=True)

    def _update_model_global_forecasts(self):
        self.model.global_forecast_df['price_Q05'] = self.usd_fil_exchange_df['price_Q05'].values
        self.model.global_forecast_df['price_Q25'] = self.usd_fil_exchange_df['price_Q25'].values
        self.model.global_forecast_df['price_Q50'] = self.usd_fil_exchange_df['price_Q50'].values
        self.model.global_forecast_df['price_Q75'] = self.usd_fil_exchange_df['price_Q75'].values
        self.model.global_forecast_df['price_Q95'] = self.usd_fil_exchange_df['price_Q95'].values

    def step(self):
        # nothing to do b/c all predictions are made upfront
        pass

#     def __init__(self,
#                  dt:float=1.,
#                  historicalFrom:datetime=None):
#         '''
        

#         Parameters
#         ----------
#         dt : float, optional
#             delta t, in days. The default is 1..
#         historicalFrom : datetime, optional
#             when to get the historical information form. The default is None,
#             in which case we download all available daily data

#         Returns
#         -------
#         None.

#         '''
        
#         self.dt=dt
#         self.currentStep=0
#         self.today=datetime.datetime.today()
#         self.historicalFrom=historicalFrom
#         self._getHistorical()

#         self.currentPrice=self.price.iloc[-1]
#         self.drift,self.vol=self._getParams()

#     def update(self):
#         'updates the price time series by one day using the GBM'
#         self.currentStep+=1
#         new_time=self.price.index[-1]+datetime.timedelta(days=1)
#         dt=self.dt
#         mu=self.drift
#         sigma=self.vol
#         logUpdate=(mu-0.5*sigma**2.0)*dt+sigma*dt**0.5*np.random.standard_normal()
#         new_price=self.price.iloc[-1]*np.exp(logUpdate)
#         self.price.loc[new_time]=new_price
        
#     def _getHistorical(self):
#         'gets historical FIL prices from Yahoo finance. This can be easily modified for other tokens'
#         print(' 📈 getting historical prices ...')
#         self.price=yf.download('FIL-USD')['Close']

#         if self.historicalFrom is not None:
#             self.price=self.price[self.price.index>self.historicalFrom]
#         self.price.rename('price')
#     def _getParams(self):
#         '''
#         computes drift (mu) and volatility (sigma) from the GBM model

#         Returns
#         -------
#         mu : float
#             drift.
#         sigma : float
#             volatility.

#         '''
#         print('🤓 computing parameters ...')

#         lp=np.log(self.price)
#         r=np.diff(lp)
        
#         sigma=np.std(r)/self.dt**0.5
#         mu=np.mean(r)/self.dt+0.5*sigma**2
        
#         return mu,sigma
        

# if __name__=='__main__':
    
#     print('💻 Testing the code! ')
    
#     import matplotlib.pyplot as plt

#     START='2022-01-01'
#     process=priceProcess(historicalFrom=START)
#     N=365
#     from tqdm import tqdm
#     import copy
#     [process.update() for _ in tqdm(range(N))]
#     plt.plot(process.price[process.price.index<process.today],label='historical price')
#     plt.plot(process.price[process.price.index>process.today],label='simulated price')
#     plt.vlines(process.today,ymin=0,ymax=process.price.max(),color='black',linestyles='dashed',label='today')
#     plt.legend()
#     plt.title('FIL price vs time')
#     plt.ylabel('Price (USD)')
#     plt.xlabel('Date')
#     plt.xticks(rotation=45)

#     plt.show()        
    
#     plt.plot(process.price[process.price.index<process.today],label='historical price')
#     Nmc=100
#     processOriginal=priceProcess(historicalFrom=START)
#     print('🎲 Now does the Monte Carlo simulation...')

#     for i in tqdm(range(Nmc)):
#         process=copy.deepcopy(processOriginal)
#         [process.update() for j in range(N)]
#         if i==0:
#             plt.plot(process.price[process.price.index>process.today],label='simulated price',alpha=0.2,color='C1')
#         else:
#             plt.plot(process.price[process.price.index>process.today],alpha=0.2,color='C1')
#     plt.vlines(process.today,ymin=0,ymax=process.price.max(),color='black',linestyles='dashed',label='today')
#     plt.xticks(rotation=45)
#     plt.ylabel('Price (USD)')
#     plt.xlabel('Date')
#     plt.legend()
#     plt.title('FIL price vs time, Monte Carlo price simulation')
#     plt.show()            
        
        