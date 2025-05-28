import numpy as np
from matplotlib.image import imread
import pandas as pd
from methods import *
from load_data import get_scores
import gridstatusio as gs
from main import write_results
from eval import eval
import time 

PREREGISTERED_METHODS = [
        'aci_fixed' , 
        'aci_decaying',
        'scalar_qt_fixed', 
        'scalar_qt_decaying', 
        'linear_qt_fixed', 
       # 'linear_batched_2_qt_fixed', 
       # 'linear_batched_4_qt_fixed',
       # 'linear_batched_8_qt_fixed',
       # 'linear_batched_16_qt_fixed',
       # 'linear_batched_32_qt_fixed',
        'linear_qt_decaying', 
       # 'linear_batched_2_qt_decaying', 
       # 'linear_batched_4_qt_decaying',
       # 'linear_batched_8_qt_decaying',
       # 'linear_batched_16_qt_decaying',
       # 'linear_batched_32_qt_decaying',
        'conformal_pi', 
        'conformal_pi_fixed', 
        'conformal_pi_decaying',  
       # 'conformal_pid_ar_scorecaster', 
       # 'conformal_pid_ar_scorecaster_fixed', 
       # 'conformal_pid_ar_scorecaster_decaying', 
       # 'conformal_pid_theta_scorecaster', 
       # 'conformal_pid_theta_scorecaster_fixed', 
       # 'conformal_pid_theta_scorecaster_decaying', 
       # 'qt_on_ar_fixed',
       # 'qt_on_ar_decaying',
    ]


if __name__ == "__main__":
    API_KEY = '1a692d6abfa547bfb58911dd29a3f088'
    START_TIME = '2024-12-18'
    END_TIME = '2025-01-04'

    # Collect scores via API.
    client = gs.GridStatusClient(API_KEY)
    df_data = client.get_dataset(dataset='ercot_load', start=START_TIME, end=END_TIME)
    time.sleep(1) # To avoid API rate limit hit.
    df_forecasts = client.get_dataset(dataset='ercot_load_forecast', start=START_TIME, end=END_TIME)
    df_merged = pd.merge(df_data, df_forecasts, on="interval_start_utc", how="inner")
    scores = np.abs(df_merged['load'] - df_merged['load_forecast']).values[200:] # Skipping first 200 to account for pre-registration time of day. 
    data_name = f'ercot_{START_TIME}_{END_TIME}'
    methods_to_run = PREREGISTERED_METHODS


    experiment_name = 'reproduced_experiments'
    data_name = f'ercot_{START_TIME}_{END_TIME}'
    val_start = '2024-12-18'
    val_end = '2025-01-04'
    hparam_file = None #f'./{experiment_name}/results/ercot_{val_start}_{val_end}.json'
    just_tune_hypers = False


    # Not using hparam file - tuning on test for baselines. Preregistered hparams for our algorithms are hard-coded in main.py
    write_results(data_name, experiment_name, methods_to_run, tune_on_val=False, val_split=0, get_thetas=False, scores=scores, add_to_saved_results=True, hparam_file=hparam_file)
    eval(data_name, experiment_name, scores=scores, write_latex=False)


