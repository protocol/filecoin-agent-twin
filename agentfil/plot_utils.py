import os
import pandas as pd
import glob

import matplotlib.pyplot as plt

def converttoMFIL(x): return x/1e6


def x_post_fn(x_ser_in):
    x_ser_in = pd.to_datetime(x_ser_in)
    return x_ser_in

# plotting helper functions & variables
def plot_experiments(
        experiment_names, results_root_dir, baseline_filecoin_df, 
        keys, exp_dirs, 
        x_post_process=None, 
        y_post_process=None,
        min_date = None, max_date = None,
        x_key='date', labels=None, plot_kwargs_list=None,
        baseline_relative=False):
    for ii, e in enumerate(exp_dirs):
        exp_name = experiment_names[ii]
        if labels is not None:
            label_str = labels[ii]
        else:
            label_str = ''
        plot_kwargs = {} if plot_kwargs_list is None else plot_kwargs_list[ii]
        plot_experiment(
            results_root_dir, baseline_filecoin_df, 
            keys, e,
            x_post_process=x_post_process, 
            y_post_process=y_post_process,
            x_key=x_key, 
            label_str=label_str,
            min_date = min_date, max_date = max_date,
            plot_kwargs=plot_kwargs,
            baseline_relative=baseline_relative)
        

def plot_experiment(
        results_root_dir, baseline_filecoin_df, 
        keys, experiment_dir,
        x_post_process=None, 
        y_post_process=None,
        x_key='date', label_str='',
        min_date = None, max_date = None,
        plot_kwargs=None,
        baseline_relative=False):
    if x_post_process is None:
        x_post_fn = lambda x: x
    else:
        x_post_fn = x_post_process
    if y_post_process is None:
        y_post_fn = lambda x: x
    else:
        y_post_fn = y_post_process
    
    # read filecoin_df into memory
    filecoin_df = pd.read_csv(os.path.join(results_root_dir, experiment_dir, 'filecoin_df.csv'))
    if min_date is not None:
        filecoin_df = filecoin_df[pd.to_datetime(filecoin_df['date']) >= pd.to_datetime(min_date)]
        baseline_filecoin_df_subset = baseline_filecoin_df[pd.to_datetime(baseline_filecoin_df['date']) >= pd.to_datetime(min_date)]
    if max_date is not None:
        filecoin_df = filecoin_df[pd.to_datetime(filecoin_df['date']) <= pd.to_datetime(max_date)]
        baseline_filecoin_df_subset = baseline_filecoin_df[pd.to_datetime(baseline_filecoin_df['date']) <= pd.to_datetime(max_date)]
      
    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
    if len(keys)==1:
        k = keys[0]
        x = x_post_fn(filecoin_df[x_key])
        y = y_post_fn(filecoin_df[k])
        if not baseline_relative:
            plt.plot(x, y, label=label_str, **plot_kwargs)
        else:
            y_baseline = y_post_fn(baseline_filecoin_df_subset[k])
            plt.plot(x, y/y_baseline*100, label=label_str, **plot_kwargs)
    else:
        # get all keys and call the combine function
        key_data = {}
        for k in keys:
            key_data[k] = filecoin_df[k]
        x = x_post_fn(filecoin_df[x_key])
        y = y_post_fn(key_data)
        if not baseline_relative:
            plt.plot(x, y, label=label_str, **plot_kwargs)
        else:
            y_baseline = y_post_fn(baseline_filecoin_df_subset[k])
            plt.plot(x, y/y_baseline*100, label=label_str, **plot_kwargs)
    plt.xticks(rotation=60)

def plot_agent(
        results_root_dir,
        keys, experiment_dir,
        x_post_process=None, 
        y_post_process=None,
        agent_ids_to_plot=None,
        df_name='agent_info_df',
        x_key='date', label_prepend='', label_postpend='', 
        min_date = None, max_date = None,
        per_agent_label_list = None,
        plot_kwargs_list=None):
    if x_post_process is None:
        x_post_fn = lambda x: x
    else:
        x_post_fn = x_post_process
    if y_post_process is None:
        y_post_fn = lambda x: x
    else:
        y_post_fn = y_post_process
    
    # read all of the agent infos into memory
    agent2agentinfo = {}
    agent2accountinginfo = {}
    flist = glob.glob(os.path.join(results_root_dir, experiment_dir, '*.csv'))
    for fp in flist:
        fname = os.path.basename(fp)
        if 'filecoin_df' in fname:
            pass
        else:
            # parse the agent # from teh filename
            agent_num = int(fname.split('_')[1])
            
            if 'agent_info' in fname:
                agent2agentinfo[agent_num] = pd.read_csv(fp)
            elif 'accounting' in fname:
                agent2accountinginfo[agent_num] = pd.read_csv(fp)
    num_total_agents = len(agent2agentinfo)
    for ii in range(num_total_agents):
        if agent_ids_to_plot is None or ii in agent_ids_to_plot:  # assumes ii is same as agent-id
            if 'agent_info' in df_name:
                agent_df = agent2agentinfo[ii]
            elif 'accounting' in df_name:
                agent_df = agent2accountinginfo[ii]
            else:
                print(df_name)
                raise ValueError("")
            
            if min_date is not None:
                agent_df = agent_df[pd.to_datetime(agent_df['date']) >= pd.to_datetime(min_date)]
            if max_date is not None:
                agent_df = agent_df[pd.to_datetime(agent_df['date']) <= pd.to_datetime(max_date)]
            
            if per_agent_label_list is None:
                l = label_prepend + '-Agent%d' % (ii,) + '-' + label_postpend
            else:
                l = label_prepend + ' %s ' % (per_agent_label_list[ii],) + label_postpend
            plot_kwargs = plot_kwargs_list[ii] if plot_kwargs_list is not None else {}
            if len(keys)==1:
                k = keys[0]
                x = x_post_fn(agent_df[x_key])
                y = y_post_fn(agent_df[k])
                plt.plot(x, y, label=l, **plot_kwargs)
            else:
                # get all keys and call the combine function
                key_data = {}
                for k in keys:
                    key_data[k] = agent_df[k]
                x = x_post_fn(agent_df[x_key])
                y = y_post_fn(key_data)
                plt.plot(x, y, label=l, **plot_kwargs)
    plt.xticks(rotation=60)
    
def x_post_fn(x_ser_in):
    x_ser_in = pd.to_datetime(x_ser_in)
    return x_ser_in