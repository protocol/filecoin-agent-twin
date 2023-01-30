#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This modeule contains some useful functions to obtain miner utility. 


Created on Fri Jan 27 11:33:03 2023

@author: juan.madrigalcianci@gmail.com
"""

import numpy as np
import requests
import pandas as pd
import json
import datetime


def getHistoricalPowers(date:str=None,
                        positiveQAP:bool=True)->pd.core.frame.DataFrame:
    '''
    Obtains the powers (QAP and RBP) of all miners at some given date. 
    If no date is provided, then it returns these values at the current date.

    Parameters
    ----------
    date : str, optional.
        Date in Y-M-D formart. The default is None.
    positiveQAP: bool, optional. 
        Returns only those miners with QAP>0. The default is True

    Returns
    -------
    miners: pandas data frame
        dataframe with columns 'miner_id', 'RBP', 'QAP'.

    '''
    
    if date is None:
        date=datetime.datetime.today().strftime('%Y-%m-%d')
    
    # this is necessary for the requests
    url = f"https://api.spacescope.io/v2/storage_provider/power?state_date={date}"
    
    payload={}
    headers = {
      'authorization': 'Bearer ghp_xJtTSVcNRJINLWMmfDangcIFCjqPUNZenoVe'
    }
    #gets data from spacescope's json endpoint API 
    response = requests.request("GET", url, headers=headers, data=payload)
    #puts everything together
    miners=pd.DataFrame(json.loads(response.text)['data'])
    miners=miners.drop(columns='stat_date')
    miners.columns=['miner_id', 'RBP', 'QAP']
    if positiveQAP:
        miners=miners[miners['QAP']>0]
    return miners


def groupMiners(miners:pd.core.frame.DataFrame,N:int,
                method:str='random')->pd.core.frame.DataFrame:
    '''
    Classifies miners into N different groups. This creates N different 
    "grouped miners", with QAP grouped miner i =sum(QAP miners in group i), 
    and similarly for RBP. This is done so that the agent based model doesn't have to 
    deal with each invdividual miner (around 4K on Jan 27)'

    Parameters
    ----------
    miners : pd.core.frame.DataFrame
        dataframe with miner information. It should contain columns RBQ and QAP. 
        This is the output of the `getHistoricalPowers` function int his same module. 
    N : int
        number of groups. This will be the number of agents in the sim
    method: str
        How to group miners. Default is to group them randomly. They can also be grouped
        by RBP  or QAP.

    Returns
    -------
    grouped_miners : pandas data frame
       dataframe with N different miners, grouped according to 'method'

    '''
    # assigns miner a a label or group, randomly
    if method=='random':
        miners['group']=np.random.randint(0,N,len(miners))
    elif method=='QAP' or method=='RBP':
        miners['group']=pd.qcut(miners[method],N,labels=False)
    else:
        raise Exception('method has to be one of "random", "QAP" or "RBP"')

        
    grouped_miners=miners.groupby(by='group')['RBP','QAP'].sum()
    return grouped_miners


if __name__=='__main__':
    DATE='2023-01-04'
    import matplotlib.pyplot as plt

    N_GROUPS=10
    miners=getHistoricalPowers(date=DATE)
    
    
    METHODS=['random','RBP','QAP','TEST']
    for m in METHODS:
        groupedMiners=groupMiners(miners=miners, N=N_GROUPS,method=m)
        groupedMiners.plot.bar()
        plt.title('grouped by '+m)
        plt.show()
    
    

