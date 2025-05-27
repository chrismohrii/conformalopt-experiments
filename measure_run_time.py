import numpy as np
from matplotlib.image import imread
import pandas as pd
from methods import *
from load_data import get_scores
import gridstatusio as gs
from main import write_results
from eval import eval
import time 
import json

PREREGISTERED_METHODS = [
    'aci_fixed' , 
    'scalar_qt_fixed', 
    'linear_qt_fixed', 
    'linear_qt_decaying'
    'linear_batched_2_qt_fixed', 
    'linear_batched_4_qt_fixed',
    'linear_batched_8_qt_fixed',
    'linear_batched_16_qt_fixed',
    'linear_batched_32_qt_fixed',
    'conformal_pi_fixed', 
    'conformal_pid_ar_scorecaster_fixed', 
    'conformal_pid_theta_scorecaster_fixed', 
    'qt_on_ar_fixed',
]

IMPLEMENTATION_MAP = {
    'aci_fixed': aci , 
    'aci_decaying': aci,
    'scalar_qt_fixed': get_scalar_qt_predictions, 
    'scalar_qt_decaying': get_scalar_qt_predictions, 
    'linear_qt_fixed': get_linear_qt_predictions, 
    'linear_batched_2_qt_fixed': get_linear_batched_qt_predictions, 
    'linear_batched_4_qt_fixed': get_linear_batched_qt_predictions, 
    'linear_batched_8_qt_fixed': get_linear_batched_qt_predictions, 
    'linear_batched_16_qt_fixed': get_linear_batched_qt_predictions, 
    'linear_batched_32_qt_fixed': get_linear_batched_qt_predictions, 
    'linear_qt_decaying': get_linear_qt_predictions, 
    'linear_batched_2_qt_decaying': get_linear_batched_qt_predictions,  
    'linear_batched_4_qt_decaying': get_linear_batched_qt_predictions, 
    'linear_batched_8_qt_decaying': get_linear_batched_qt_predictions, 
    'linear_batched_16_qt_decaying': get_linear_batched_qt_predictions, 
    'linear_batched_32_qt_decaying': get_linear_batched_qt_predictions, 
    'conformal_pi': get_pi_control, 
    'conformal_pi_fixed': get_pi_control, 
    'conformal_pi_decaying': get_pi_control, 
    'conformal_pid_ar_scorecaster': get_pid_control,
    'conformal_pid_ar_scorecaster_fixed': get_pid_control,
    'conformal_pid_ar_scorecaster_decaying': get_pid_control,
    'conformal_pid_theta_scorecaster': get_pid_control, 
    'conformal_pid_theta_scorecaster_fixed': get_pid_control, 
    'conformal_pid_theta_scorecaster_decaying': get_pid_control, 
    'qt_on_ar_fixed': qt_on_ar,
    'qt_on_ar_decaying': qt_on_ar,
}


if __name__ == "__main__":
    np.random.seed(0)
    API_KEY = '1a692d6abfa547bfb58911dd29a3f088'
    START_TIME = '2024-12-18'
    END_TIME = '2025-01-04'

    # Collect scores via API.
    client = gs.GridStatusClient(API_KEY)
    df_data = client.get_dataset(dataset='ercot_load', start=START_TIME, end=END_TIME)
    time.sleep(1) # To avoid API rate limit hit.
    df_forecasts = client.get_dataset(dataset='ercot_load_forecast', start=START_TIME, end=END_TIME)
    df_merged = pd.merge(df_data, df_forecasts, on="interval_start_utc", how="inner")
    scores = np.abs(df_merged['load'] - df_merged['load_forecast']).values
    data_name = f'ercot_{START_TIME}_{END_TIME}'
    methods_to_run = PREREGISTERED_METHODS

    experiment_name='preregistered_w_val'
    data_name = f'ercot_{START_TIME}_{END_TIME}'
    val_start = '2024-12-02'
    val_end = '2024-12-16'
    hparam_file = f'./{experiment_name}/results/ercot_{val_start}_{val_end}.json'
    methods_to_run = PREREGISTERED_METHODS

    scores = get_scores('elec')

    with open(hparam_file, 'r') as json_file:
        methods = json.load(json_file)

    val_etas = np.ones(len(scores))

    methods['qt_on_ar_fixed']['kwargs']['window_size'] = 200
    times = []

    max_time = 0

    for method_name, method_details in methods.items():
        if method_name in ['qt_on_ar_fixed']:
            time_start = time.time()

            # Code to measure
            predictions = IMPLEMENTATION_MAP[method_name](scores, 0, val_etas, 0.5, **method_details['kwargs'])

            # End timing
            time_end = time.time()

            # Calculate elapsed time
            elapsed_time = time_end - time_start
            times.append((method_name, elapsed_time))
            if elapsed_time > max_time:
                max_time = elapsed_time


    print(times)

    times = [(method, time/max_time) for (method, time) in times]

    print(times)
