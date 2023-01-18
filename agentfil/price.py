#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the FIL price module.
It downloads historical data for FIL token price, extracts info, and simulates
the token price based on a geometric Brownian motion

Do your own research. This is not to be considered as any sort of financial advice.


requirements:
    -numpy
    -yfinace (pip install yahoo finance. This is done to obtain historical data)
    


@author: juan.madrigalcianci
"""

import numpy as np
import yfinance as yf
import datetime
class priceProcess:
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
    def __init__(self,
                 dt:float=1.,
                 historicalFrom:datetime=None):
        '''
        

        Parameters
        ----------
        dt : float, optional
            delta t, in days. The default is 1..
        historicalFrom : datetime, optional
            when to get the historical information form. The default is None,
            in which case we download all available daily data

        Returns
        -------
        None.

        '''
        
        self.dt=dt
        self.currentStep=0
        self.today=datetime.datetime.today()
        self.historicalFrom=historicalFrom
        self._getHistorical()

        self.currentPrice=self.price.iloc[-1]
        self.drift,self.vol=self._getParams()

    def update(self):
        'updates the price time series by one day using the GBM'
        self.currentStep+=1
        new_time=self.price.index[-1]+datetime.timedelta(days=1)
        dt=self.dt
        mu=self.drift
        sigma=self.vol
        logUpdate=(mu-0.5*sigma**2.0)*dt+sigma*dt**0.5*np.random.standard_normal()
        new_price=self.price.iloc[-1]*np.exp(logUpdate)
        self.price.loc[new_time]=new_price
        
    def _getHistorical(self):
        'gets historical FIL prices from Yahoo finance. This can be easily modified for other tokens'
        print(' 📈 getting historical prices ...')
        self.price=yf.download('FIL-USD')['Close']

        if self.historicalFrom is not None:
            self.price=self.price[self.price.index>self.historicalFrom]
        self.price.rename('price')
    def _getParams(self):
        '''
        computes drift (mu) and volatility (sigma) from the GBM model

        Returns
        -------
        mu : float
            drift.
        sigma : float
            volatility.

        '''
        print('🤓 computing parameters ...')

        lp=np.log(self.price)
        r=np.diff(lp)
        
        sigma=np.std(r)/self.dt**0.5
        mu=np.mean(r)/self.dt+0.5*sigma**2
        
        return mu,sigma
        

if __name__=='__main__':
    
    print('💻 Testing the code! ')
    
    import matplotlib.pyplot as plt

    START='2022-01-01'
    process=priceProcess(historicalFrom=START)
    N=365
    from tqdm import tqdm
    import copy
    [process.update() for _ in tqdm(range(N))]
    plt.plot(process.price[process.price.index<process.today],label='historical price')
    plt.plot(process.price[process.price.index>process.today],label='simulated price')
    plt.vlines(process.today,ymin=0,ymax=process.price.max(),color='black',linestyles='dashed',label='today')
    plt.legend()
    plt.title('FIL price vs time')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    plt.show()        
    
    plt.plot(process.price[process.price.index<process.today],label='historical price')
    Nmc=100
    processOriginal=priceProcess(historicalFrom=START)
    print('🎲 Now does the Monte Carlo simulation...')

    for i in tqdm(range(Nmc)):
        process=copy.deepcopy(processOriginal)
        [process.update() for j in range(N)]
        if i==0:
            plt.plot(process.price[process.price.index>process.today],label='simulated price',alpha=0.2,color='C1')
        else:
            plt.plot(process.price[process.price.index>process.today],alpha=0.2,color='C1')
    plt.vlines(process.today,ymin=0,ymax=process.price.max(),color='black',linestyles='dashed',label='today')
    plt.xticks(rotation=45)
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.title('FIL price vs time, Monte Carlo price simulation')
    plt.show()            
        
        