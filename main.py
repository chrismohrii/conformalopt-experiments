import json
import numpy as np
from tqdm import tqdm
import os
from methods import *
from load_data import get_scores
from eval import eval
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import pacf
import sys

# Maps method name to implementation. Order is important.
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
    'conformal_pid_theta_scorecaster': get_pid_control, 
    'conformal_pid_theta_scorecaster_fixed': get_pid_control, 
    'conformal_pid_theta_scorecaster_decaying': get_pid_control, 
    'conformal_pid_ar_scorecaster': get_pid_control,
    'conformal_pid_ar_scorecaster_fixed': get_pid_control,
    'conformal_pid_ar_scorecaster_decaying': get_pid_control,
    'qt_on_ar_fixed': qt_on_ar,
    'qt_on_ar_decaying': qt_on_ar,
}


def find_prefix(strings, target):
    for s in strings:
        if target.startswith(s):
            return s
    return None  # if no match is found

# Function to convert ndarray to list
def convert_ndarray_to_list(d):
    if isinstance(d, dict):
        return {k: convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_ndarray_to_list(i) for i in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()  # Convert ndarray to list
    else:
        return d


# From https://arxiv.org/abs/2402.01139
def smooth_array(arr, window_size):
    # Create a window of ones of length window_size
    window = np.ones(window_size) / window_size
    
    # Use convolve to apply the window to the array
    # 'valid' mode returns output only where the window fits completely
    smoothed = np.convolve(arr, window, mode='valid')
    
    return smoothed

def init_methods_dict(data_name, p_order=1, T_burnin=20):
    """
    Initializes a dictionary of online conformal methods with their respective parameters. 

    Parameters:
    data_name (str): The name of the dataset to be used.
    p_order (int, optional): The order of the autoregressive model used in any of the methods. Default is 1.
    T_burnin (int, optional): Burn in time period, used in methods from conformal PID paper.

    Returns:
    dict: A dictionary containing the configuration for each method. Each key in the dictionary represents a method and its value is another dictionary with the following keys:
        - lr_type: The type of learning rate to be used ('fixed' or 'decaying'). For PID methods this does not matter.
        - fitted_lr: The fitted learning rate. This is initially set to None and is expected to be updated during the execution of the method.
        - kwargs: A dictionary of additional arguments to be passed to the method.
    """
    Csat = None 
    KI = None 
    p_order = 0 
    p_order_for_ar_scorecaster = 1 # For PID method.
    pid_window_size = 1000000 if 'residual' in data_name else 200 # only restrict for large data.
    qt_on_ar_window_size = 200 if not 'ercot' in data_name else 10 # Always 200, but put 10 in ercot preregistration.

    methods =  {
        'aci_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'T_burnin': T_burnin,
                'window_size': 100000, # Copying PID paper.
            },
        },
        'aci_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'T_burnin': T_burnin,
                'window_size': 100000, # Copying PID paper.
            },
        },
        'scalar_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {},
        },
        'scalar_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {},
        },
        'linear_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
            },
        },
        'linear_batched_2_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 2,
            },
        },
        'linear_batched_4_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,                
                'batch_size': 4,
            },
        },
        'linear_batched_8_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 8,
            },
        },
        'linear_batched_16_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 16,
            },
        },
        'linear_batched_32_qt_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 32,
            },
        },
        'linear_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
            },
        },
        'linear_batched_2_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 2,
         },
        },
        'linear_batched_4_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 4,
         },
        },
        'linear_batched_8_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 8,
         },
        },
        'linear_batched_16_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 16,
         },
        },
        'linear_batched_32_qt_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'bias': None,
                'batch_size': 32,
         },
        },
        'conformal_pi': {
            'lr_type': 'fixed', # Irrelevant due to proportional_lr: True setting below.
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'proportional_lr': True
            }
        },
        'conformal_pi_fixed': {
            'lr_type': 'fixed', 
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'proportional_lr': False
            }
        },
        'conformal_pi_decaying': {
            'lr_type': 'decaying', 
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'proportional_lr': False
            }
        },
        'conformal_pid_theta_scorecaster': {
            'lr_type': 'fixed', # Irrelevant due to proportional_lr: True setting below.
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'scorecaster': 'theta',
                'proportional_lr': True,
                'window_size': pid_window_size,
            }
        },
        'conformal_pid_theta_scorecaster_fixed': {
            'lr_type': 'fixed', 
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'scorecaster': 'theta',
                'proportional_lr': False,
                'window_size': pid_window_size,                
            }
        },
        'conformal_pid_theta_scorecaster_decaying': {
            'lr_type': 'decaying', 
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'scorecaster': 'theta',
                'proportional_lr': False,
                'window_size': pid_window_size,
            }
        },
        'conformal_pid_ar_scorecaster': {
            'lr_type': 'fixed', # Irrelevant due to proportional_lr: True setting below.
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'scorecaster': 'ar',
                'p_order_for_ar_scorecaster': p_order_for_ar_scorecaster,
                'proportional_lr': True,
                'window_size': pid_window_size,
            }
        },
        'conformal_pid_ar_scorecaster_fixed': {
            'lr_type': 'fixed', 
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'scorecaster': 'ar',
                'p_order_for_ar_scorecaster': p_order_for_ar_scorecaster,
                'proportional_lr': False,
                'window_size': pid_window_size,
            }
        },
        'conformal_pid_ar_scorecaster_decaying': {
            'lr_type': 'decaying', 
            'fitted_lr': None,
            'kwargs': {
                'Csat': Csat,
                'KI': KI,
                'ahead': 1,
                'T_burnin': T_burnin,
                'scorecaster': 'ar',
                'p_order_for_ar_scorecaster': p_order_for_ar_scorecaster,
                'proportional_lr': False,
                'window_size': pid_window_size,
            }
        },
        'qt_on_ar_fixed': {
            'lr_type': 'fixed',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'window_size': qt_on_ar_window_size,
            },
        },
        'qt_on_ar_decaying': {
            'lr_type': 'decaying',
            'fitted_lr': None,
            'kwargs': {
                'p_order': p_order,
                'window_size': qt_on_ar_window_size,
            },
        }
    }

    for method in methods:
        if method.startswith('linear') or method.startswith('qt_on_ar'):
            methods[method]['tuning_set'] = 'val'
        else:
            methods[method]['tuning_set'] = 'test'

    return methods


def write_results(data_name, experiment_name, methods_to_run=None, tune_on_val=True, get_thetas = False, scores=None, alpha=0.1, epsilon=0.1, val_split=0.25, hparam_file=None, add_to_saved_results=False):
    if hparam_file is not None:
        with open(hparam_file, 'r') as json_file:
            methods_all = json.load(json_file)
            #print(methods_all)

    methods_all = init_methods_dict(data_name) if hparam_file is None else methods_all
    if methods_to_run:
        methods = {key: methods_all[key] for key in methods_to_run if key in methods_all}
    else:
        methods = methods_all

    scores = get_scores(data_name, scores=scores)
    print(f'{data_name} has length {len(scores)}')

    q_1 = 1 if not 'ercot' in data_name else scores[0]
    cov_gap = 0.01
    # Split into validation and test. 
    val_size = int(val_split * len(scores))
    test_scores = scores[val_size:] 
    true_val_scores = scores[:val_size] if tune_on_val else test_scores # Val scores don't matter if hparams are provided.
    T_val = len(true_val_scores)
    T_test = len(test_scores)

    print(f'T_val: {T_val}; T_test: {T_test}')

    for method_name, method_details in methods.items():
        print(f'Running {method_name}...')

        #if 'fixed' in method_name:
        #    cov_gap=0.02

        # 1. Tune hyperparameters on validation set, unless hparams are provided.
        if hparam_file is None:
            # If this is a baseline method, cheat by setting the validation set
            # to the test set.
            val_scores = true_val_scores if method_details['tuning_set'] == 'val' else test_scores
            T_val = len(val_scores)
            # lrs = [1e-3] if method_name=='linear_qt_fixed' else [1e-1]# [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5] 

            lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]  #[5e-3] if 'aci' in method_name else [1e-1]# [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5] 
            
            if False:
                if 'pi' in method_name:
                    lrs = [0.1]
                if 'aci' in method_name:
                    lrs = [0.005]

            # For synth plots. lrs = [1e-3] if method_name=='linear_qt_fixed' else [1e-1]# [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5] 

            if not ("Csat" in method_details['kwargs'] and "KI" in method_details['kwargs']):
                if "p_order" in method_details['kwargs']: 
                    # Our methods.
                    best_quantile_loss = np.inf
                    best_lr = None
                    best_p_order = None
                    best_bias = None

                    coverage_achieved = False
                    best_quantile_loss_coverage_constrained = np.inf
                    best_lr_coverage_constrained = None
                    best_p_order_coverage_constrained = None
                    best_bias_coverage_constrained = None

                    if 'ercot' in data_name:
                        if method_name == 'linear_qt_fixed':
                            lrs = [1e-5] 
                        elif 'qt_fixed' in method_name:
                            lrs = [1e-4]
                        elif 'qt_decaying' in method_name:
                            lrs = [1e-3] 
                        elif method_name == 'qt_on_ar_fixed':
                            lrs = [10.0]
                        elif method_name == 'qt_on_ar_decaying':
                            lrs = [100.0]
                        
                    for lr in tqdm(lrs):
                        a, _ = pacf(val_scores, nlags=20, alpha=0.05)
                        #print(a)
                    # print((a, np.abs(a) >= 0.1, sum(np.abs(a) >= 0.3) - 1))

                        p_orders = [min(int(np.argmax(np.abs(a) <= 0.2) - 1 if np.any(np.abs(a) <= 0.2) else len(a) - 1), 2)] #[int(sum(np.abs(a) >= 0.2) - 1)]#[0, 1, 2, 3, 10, 20]#[0, 1, 2]#int(sum(np.abs(a) >= 0.3) - 1)
                        p_orders = [0, 1, 2] if not 'ercot' in method_name else [1] #[0, 1, 2, 5, 10, 20]
                        print(f'p_orders: {p_orders}')
                        for p_order in p_orders: # p_orders has length 1
                            method_details['kwargs']["p_order"] = p_order

                            biases = [1e-1, 1e0, 5, 1e1, 200, 1e2, 1e3] if method_name.startswith('linear') else [0] #[1e-0, 5]# #[1e-1, 1e0, 5, 1e1,  1e2, 1e3] #[200] #[] if method_name.startswith('linear') else [0]
                            if 'ercot' in method_name and method_name.startswith('linear'):
                                biases = [200]
                            for bias in biases:
                                if method_name.startswith('linear'):
                                    method_details['kwargs']["bias"] = bias
                                # Get learning rate schedule.
                                if method_details['lr_type'] == 'fixed':
                                    val_etas = lr * np.ones(T_val)
                                elif method_details['lr_type'] == 'decaying':
                                    val_etas = lr * np.array([1/(t**(1/2+epsilon)) for t in range(1, T_val+1)])
                                else:
                                    raise ValueError(f'Unknown learning rate type: {method_details["lr_type"]}')

                                # Get val predictions with candidate lr
                                predictions = IMPLEMENTATION_MAP[method_name](val_scores, q_1, val_etas, alpha, **method_details['kwargs'])
                                curr_quantile_loss = quantile_loss(val_scores, predictions, 1-alpha)
                                curr_run_coverage_achieved = np.abs(np.mean(predictions >= val_scores) - (1 - alpha)) <= cov_gap

                                if not coverage_achieved and curr_run_coverage_achieved:
                                    coverage_achieved = True

                                if curr_quantile_loss < best_quantile_loss:
                                    best_quantile_loss = curr_quantile_loss
                                    best_lr = lr
                                    best_p_order = p_order
                                    best_bias = bias
                                    print(f'Associated coverage: {np.mean(predictions >= val_scores)}')

                                if curr_run_coverage_achieved and curr_quantile_loss < best_quantile_loss_coverage_constrained:
                                    best_quantile_loss_coverage_constrained = curr_quantile_loss
                                    best_lr_coverage_constrained = lr
                                    best_p_order_coverage_constrained = p_order
                                    best_bias_coverage_constrained = bias


                    method_details['fitted_lr'] = best_lr if not coverage_achieved else best_lr_coverage_constrained
                    method_details['kwargs']['p_order'] = best_p_order if not coverage_achieved else best_p_order_coverage_constrained
                    if method_name.startswith('linear'):
                        method_details['kwargs']['bias'] = best_bias if not coverage_achieved else best_bias_coverage_constrained

                    print(f'Coverage achieved: {coverage_achieved}')
                    print(f'Best lr for {method_name}: {method_details["fitted_lr"]}')
                    print(f'Best p_order for {method_name}: {method_details["kwargs"]["p_order"]}')
                    print(f'Best bias for {method_name}: {best_bias}')

                else:
                    # Basic ACI and scalar qt methods.
                    best_quantile_loss = np.inf
                    best_lr = None

                    coverage_achieved = False
                    best_quantile_loss_coverage_constrained = np.inf
                    best_lr_coverage_constrained = None

                    for lr in tqdm(lrs):
                        # Get learning rate schedule.
                        if method_details['lr_type'] == 'fixed':
                            val_etas = lr * np.ones(T_val)
                        elif method_details['lr_type'] == 'decaying':
                            val_etas = lr * np.array([1/(t**(1/2+epsilon)) for t in range(1, T_val+1)])
                        else:
                            raise ValueError(f'Unknown learning rate type: {method_details["lr_type"]}')

                        # Get val predictions with candidate lr
                        predictions = IMPLEMENTATION_MAP[method_name](val_scores, q_1, val_etas, alpha, **method_details['kwargs'])
                        curr_quantile_loss = quantile_loss(val_scores, predictions, 1-alpha)
                        curr_run_coverage_achieved = np.abs(np.mean(predictions >= val_scores) - (1 - alpha)) <= cov_gap

                        if not coverage_achieved and curr_run_coverage_achieved:
                            coverage_achieved = True

                        if curr_quantile_loss < best_quantile_loss:
                            best_quantile_loss = curr_quantile_loss
                            best_lr = lr

                        if curr_run_coverage_achieved and curr_quantile_loss < best_quantile_loss_coverage_constrained:
                            best_quantile_loss_coverage_constrained = curr_quantile_loss
                            best_lr_coverage_constrained = lr


                    method_details['fitted_lr'] = best_lr if not coverage_achieved else best_lr_coverage_constrained
                    print(f'Coverage achieved: {coverage_achieved}')
                    print(f'Best lr for {method_name}: {method_details["fitted_lr"]}')

            else:
                # Conformal PID paper methods.
                best_quantile_loss = np.inf
                best_lr = None
                best_Csat = None
                best_KI = None

                coverage_achieved = False
                best_quantile_loss_coverage_constrained = np.inf
                best_lr_coverage_constrained = None
                best_Csat_coverage_constrained = None
                best_KI_coverage_constrained = None

                delta = 0.01
                # From Appendix B of PID paper.
                Csat_suggested = (2/np.pi)*(np.ceil(np.log(T_val)*delta) - (1/np.log(T_val)))
                KI_suggested = np.max(np.abs(scores)) # Could be cheating, but for baseline so it's fine. 

                for lr in tqdm(lrs):
                    for Csat in [0.1, 1, 5, Csat_suggested, 20]:
                        for KI in [10, 100, KI_suggested, 200, 1000]:
                            method_details['kwargs']["Csat"] = Csat
                            method_details['kwargs']["KI"] = KI
                            # Get learning rate schedule. #This doesn't matter when proportional_lr=True.
                            if method_details['lr_type'] == 'fixed':
                                val_etas = lr * np.ones(T_val)
                            elif method_details['lr_type'] == 'decaying':
                                val_etas = lr * np.array([1/(t**(1/2+epsilon)) for t in range(1, T_val+1)])
                            else:
                                raise ValueError(f'Unknown learning rate type: {method_details["lr_type"]}')

                            # Get val predictions with candidate lr
                            predictions = IMPLEMENTATION_MAP[method_name](val_scores, q_1, val_etas, alpha, **method_details['kwargs'])
                            curr_quantile_loss = quantile_loss(val_scores, predictions, 1-alpha)
                            curr_run_coverage_achieved = np.abs(np.mean(predictions >= val_scores) - (1 - alpha)) <= cov_gap

                            if not coverage_achieved and curr_run_coverage_achieved:
                                coverage_achieved = True

                            if curr_quantile_loss < best_quantile_loss:
                                best_quantile_loss = curr_quantile_loss
                                best_lr = lr
                                best_Csat = Csat
                                best_KI = KI
                                print(f'Associated coverage PI: {np.mean(predictions >= val_scores)}')

                            if curr_run_coverage_achieved and curr_quantile_loss < best_quantile_loss_coverage_constrained:
                                best_quantile_loss_coverage_constrained = curr_quantile_loss
                                best_lr_coverage_constrained = lr
                                best_Csat_coverage_constrained = Csat
                                best_KI_coverage_constrained = KI

                method_details['fitted_lr'] = best_lr if not coverage_achieved else best_lr_coverage_constrained
                method_details['kwargs']["Csat"] = best_Csat if not coverage_achieved else best_Csat_coverage_constrained
                method_details['kwargs']["KI"] = best_KI if not coverage_achieved else best_KI_coverage_constrained

                print(f'Coverage achieved: {coverage_achieved}')
                print(f'Best lr for {method_name}: {method_details["fitted_lr"]}')
                print(f'Best Csat for {method_name}: {method_details["kwargs"]["Csat"]}')
                print(f'Best KI for {method_name}: {method_details["kwargs"]["KI"]}')

        # 2. Get predictions on test set.
        if T_test > 0: # Can have no test set, just interested in tuned hparams.
            if method_details['lr_type'] == 'fixed':
                test_etas = method_details['fitted_lr'] * np.ones(T_test)
            elif method_details['lr_type'] == 'decaying':
                test_etas = method_details['fitted_lr'] * np.array([1/(t**(1/2+epsilon)) for t in range(1, T_test+1)])
            else:
                raise ValueError(f'Unknown learning rate type: {method_details["lr_type"]}')

            method_details['predictions'] = IMPLEMENTATION_MAP[method_name](test_scores, q_1, test_etas, alpha, **method_details['kwargs']) 
            #print(method_details['predictions'])
            #method_details['predictions'] = [x  for x in method_details['predictions']]
            if get_thetas and data_name.startswith('synth'):
                if 'batch' in method_name:
                    method_details['thetas'] = get_linear_batched_qt_thetas(test_scores, q_1, test_etas, alpha, **method_details['kwargs'])
                elif 'linear' in method_name:
                    method_details['thetas'] = get_linear_qt_thetas(test_scores, q_1, test_etas, alpha, **method_details['kwargs'])

            # 3. Evaluate all metrics.
            results_dict = {}
            W = 1000
            results_dict['quantile_loss'] = float(quantile_loss(test_scores, method_details['predictions'], 1-alpha))
            results_dict['set_size'] = float(np.mean(method_details['predictions']))
            results_dict['coverage'] = float(np.mean(method_details['predictions'] >= test_scores))
            results_dict['absolute_loss'] = float(absolute_loss(test_scores, method_details['predictions']))
            results_dict['square_loss'] = float(square_loss(test_scores, method_details['predictions']))
            results_dict['pos_excess'] = float(np.mean(method_details['predictions'][method_details['predictions'] > test_scores] - test_scores[method_details['predictions'] > test_scores]))
            results_dict['neg_excess'] = float(np.mean(method_details['predictions'][method_details['predictions'] < test_scores] - test_scores[method_details['predictions'] < test_scores]))
            results_dict['coverages'] = (test_scores <= method_details['predictions']).astype(int)
            results_dict[f'{W}-smoothed coverages'] = smooth_array(results_dict['coverages'], W)

            method_details['results'] = results_dict
            method_details['val_size'] = val_size
            print(f'Results for {method_name}: {results_dict}')

    # Add to saved results if they exist and desired.
    if add_to_saved_results:
        saved_results_file = f'./{experiment_name}/results/{data_name}.json'

        try:
            with open(saved_results_file, 'r') as json_file:
                saved_methods = json.load(json_file)

            # Add new results to saved file.
            for key in methods.keys():
                saved_methods[key] = methods[key]

            # Reorder keys in saved_methods to match IMPLEMENTATION_MAP
            saved_methods = {key: saved_methods[key] for key in IMPLEMENTATION_MAP if key in saved_methods}

            methods = saved_methods
        except FileNotFoundError:
            print(f"The file {saved_results_file} does not exist. Passing.")

    # Save results in JSON
    methods = convert_ndarray_to_list(methods)
        
    print('Writing results to json')
    # Write the dictionary to a JSON file.
    # Create result folder if it doesn't already exist.
    os.makedirs(f'./{experiment_name}/results', exist_ok=True)
    fname = f'./{experiment_name}/results/{data_name}.json'
    with open(fname, 'w') as json_file:
        json.dump(methods, json_file, indent=4)  # Use indent for pretty-printing
    print(f'Finished writing {fname}')


if __name__ == "__main__":
    # Miscoverage rate.
    alpha = 0.1
    # Decaying learning rates are c * t^-(1/2+\epsilon). Squared summable but not summable as assumptions require.
    epsilon = 0.1 # 0.1

    # List of methods to run. None = all methods.
    methods_to_run = [
        'aci_fixed' , 
        'aci_decaying',
        'scalar_qt_fixed', 
        'scalar_qt_decaying', 
        'linear_qt_fixed', 
    #    'linear_batched_2_qt_fixed', 
    #    'linear_batched_4_qt_fixed',
    #    'linear_batched_8_qt_fixed',
    #    'linear_batched_16_qt_fixed',
    #    'linear_batched_32_qt_fixed',
        'linear_qt_decaying', 
    #    'linear_batched_2_qt_decaying', 
    #    'linear_batched_4_qt_decaying',
    #    'linear_batched_8_qt_decaying',
    #    'linear_batched_16_qt_decaying',
    #    'linear_batched_32_qt_decaying',
        'conformal_pi', 
        'conformal_pi_fixed', 
        'conformal_pi_decaying',  
    #    'conformal_pid_ar_scorecaster', 
    #    'conformal_pid_ar_scorecaster_fixed', 
    #    'conformal_pid_ar_scorecaster_decaying', 
    #    'conformal_pid_theta_scorecaster', 
    #    'conformal_pid_theta_scorecaster_fixed', 
    #    'conformal_pid_theta_scorecaster_decaying', 
    #    'qt_on_ar_fixed',
    #    'qt_on_ar_decaying',
    ]

    experiment_name = "reproduced_experiments"

    if len(sys.argv) > 2:
        data_abbr = sys.argv[1]
        model_type = sys.argv[2]
        data_name = f'{data_abbr}_{model_type}_absolute-residual_scores' 
        write_results(data_name, experiment_name, methods_to_run, val_split=0.33, tune_on_val=True, add_to_saved_results=True)
        eval(data_name, experiment_name, write_latex=False)

    else:
        data_name = sys.argv[1]
        write_results(data_name, experiment_name, methods_to_run, val_split=0.25, tune_on_val=True, get_thetas=False, add_to_saved_results=True)
        eval(data_name, experiment_name, write_latex=False)
