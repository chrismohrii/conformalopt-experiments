import numpy as np
import pandas as pd
import time
import gridstatusio as gs


def get_scores(data_name, scores=None):
    """
    Loads and processes scores for a specified dataset. The scores are |Y_t - \hat Y_t| for 
    various base forcasters \hat Y_t, except for M4 which is just a time series. 

    Parameters:
    data_name (str): Name of the dataset. Supported options:
        - 'elec': Electricity data, base forcaster is a one-day delayed moving average.
        - 'M4-monthly', 'M4-yearly': M4 competition data, returns values from column 'V2'.
        - 'M4-monthly-log', 'M4-yearly-log': Same as above, with log transformation.
        - 'daily-climate*': Daily climate data. Base forecaster can be Theta, prophet, or Transformer models.  
        - 'AMZN*', 'GOOGL*', 'MSFT*': Stock data, applies log and mean adjustment. Base forecaster can be Theta, prophet, or Transformer models.
    Returns:
    np.ndarray: Processed score array (first 30 values skipped).
    """
    if scores is None:
        if data_name == 'elec': # length 45264
            # Score is |Y_t - \hat Y_t| where \hat Y_t is a one-day delayed moving average
            data = pd.read_csv('./data/electricity-normalized.csv')
            Y = data['nswdemand'].to_numpy()
            # Bug in PID paper code: actually predicting one-day delayed moving average now, as paper claims.
            Yhat = [np.mean(Y[i:i+24]) for i in range(len(Y[48:]))]
            Y = Y[48:]
            scores = np.abs(Y - Yhat)
        elif data_name == "M4-monthly":
            data = pd.read_csv('./data/M4/Monthly-test.csv')
            scores = data['V2'].to_numpy()[:5000]
        elif data_name == "M4-yearly":
            data = pd.read_csv('./data/M4/Yearly-test.csv')
            scores = data['V2'].to_numpy()
        elif data_name.startswith('daily-climate'):
            filename = f'./data/{data_name}.csv'
            scores = np.loadtxt(filename)
        elif data_name =="synthetic_AR_3":
            scores = np.load('data/synthetic/synthetic_AR_3_scores.npy')
        elif data_name =="synthetic_AR_2":
            scores = np.load('data/synthetic/synthetic_AR_2_scores.npy')
        elif data_name.startswith("synthetic_AR_2_1M"):

            np.random.seed(int(data_name.split('_')[-1]))
            # Parameters
            n = 1_000_000  # Length of the time series
            phi = [0.3, -0.3]  # AR(2) parameters
            sigma = 1.0  # Standard deviation of the noise

            # Generate white noise
            noise = np.random.normal(0, sigma, n)

            # Initialize the time series
            y = np.zeros(n)

            # Generate the AR(2) time series
            for t in range(2, n):
                y[t] = phi[0] * y[t-1] + phi[1] * y[t-2] + noise[t]

            scores = y 
        elif data_name.startswith("synthetic_AR_1"):

            np.random.seed(int(data_name.split('_')[-1]))
            # Parameters
            n = 5_000  # Length of the time series
            phi = [0.99]  # AR(2) parameters
            sigma = 1.0  # Standard deviation of the noise

            # Generate white noise
            noise = np.random.normal(0, sigma, n)

            # Initialize the time series
            y = np.zeros(n)

            # Generate the AR(2) time series
            for t in range(2, n):
                y[t] = phi[0] * y[t-1] + noise[t]

            scores = y 
        elif data_name =="gaussian":
            scores = []
            for i in range(10000000):
                scores.append(np.random.normal(scale = i // 10))
            scores = np.array(scores)
        elif data_name == 'ercot_preregistered':
            API_KEY = '1a692d6abfa547bfb58911dd29a3f088'
            START_TIME = '2024-12-18'
            END_TIME = '2025-01-04'

            # Collect scores via API.
            client = gs.GridStatusClient(API_KEY)
            df_data = client.get_dataset(dataset='ercot_load', start=START_TIME, end=END_TIME)
            time.sleep(1) # To avoid API rate limit hit.
            df_forecasts = client.get_dataset(dataset='ercot_load_forecast', start=START_TIME, end=END_TIME)
            df_merged = pd.merge(df_data, df_forecasts, on="interval_start_utc", how="inner")
            scores = np.abs(df_merged['load'] - df_merged['load_forecast']).values[200:]

        elif data_name =="ercot":
            scores = np.load('data/ercot_load_10000.npy') 
        elif data_name =="ercot_start":
            scores = np.load('data/ercot_load_1000_start.npy') 
        elif data_name =="ercot_500_000_end":
            scores = np.load('data/ercot_load_500_000_end.npy')[-10_000:] 
        else:
            # Stocks (AMZN, GOOGL, MSFT) routed here.
            filename = f'./data/{data_name}.csv'
            scores = np.loadtxt(filename)
            #scores = scores / max(scores)
            # We take the log as the conformal PID paper claims.
            #scores = np.log(scores) 
            #scores -= np.mean(scores)

    # Sometimes the first few scores in these datasets are nonsense. 
    return scores[30:]
