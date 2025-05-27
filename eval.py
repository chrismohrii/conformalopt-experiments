import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from methods import *
from load_data import get_scores
from methods import quantile_loss, absolute_loss
from hypothesis_testing_utils import one_sided_h_test


METRICS = ['coverage', 'quantile_loss', 'set_size', 'absolute_loss', 'square_loss', 'pos_excess', 'neg_excess']

METHOD_NAME_LEGEND_MAP = {
    'aci_fixed': 'ACI (fixed)', 
    'aci_decaying': 'ACI (decaying)',
    'scalar_qt_fixed': 'SQT (fixed)',
    'scalar_qt_decaying': 'SQT (decaying)',
    'linear_qt_fixed': 'LQT (fixed)',
    'linear_batched_2_qt_fixed': 'LQT batch 2 (fixed)', 
    'linear_batched_4_qt_fixed': 'LQT batch 4 (fixed)', 
    'linear_batched_8_qt_fixed': 'LQT batch 8 (fixed)', 
    'linear_batched_16_qt_fixed': 'LQT batch 16 (fixed)',  
    'linear_batched_32_qt_fixed': 'LQT batch 32 (fixed)', 
    'linear_qt_decaying': 'LQT (decaying)',
    'linear_batched_2_qt_decaying': 'LQT batch 2 (decaying)', 
    'linear_batched_4_qt_decaying': 'LQT batch 4 (decaying)', 
    'linear_batched_8_qt_decaying': 'LQT batch 8 (decaying)', 
    'linear_batched_16_qt_decaying': 'LQT batch 16 (decaying)', 
    'linear_batched_32_qt_decaying': 'LQT batch 32 (decaying)', 
    'conformal_pi': 'PI', 
    'conformal_pi_fixed': 'PI (fixed)', 
    'conformal_pi_decaying': 'PI (decaying)', 
    'conformal_pid_theta_scorecaster': 'PID(theta)', 
    'conformal_pid_theta_scorecaster_fixed': 'PID(theta) (fixed)', 
    'conformal_pid_theta_scorecaster_decaying': 'PID(theta) (decaying)', 
    'conformal_pid_ar_scorecaster': 'PID(AR)', 
    'conformal_pid_ar_scorecaster_fixed': 'PID(AR) (fixed)', 
    'conformal_pid_ar_scorecaster_decaying': 'PID (AR) (decaying)',
    'qt_on_ar_fixed': 'SQT+AR (fixed)',
    'qt_on_ar_decaying': 'SQT+AR (decaying)',
}

# From https://arxiv.org/abs/2402.01139
def smooth_array(arr, window_size):
    # Create a window of ones of length window_size
    window = np.ones(window_size) / window_size
    
    # Use convolve to apply the window to the array
    # 'valid' mode returns output only where the window fits completely
    smoothed = np.convolve(arr, window, mode='valid')
    
    return smoothed


def plot_pred_vs_score_window(location, window, index, predictions, scores, data_name, method_name, experiment_name):
    """
    Plots predictions versus actual scores over a given window of data points and saves the plot as an image.

    Parameters:
    - location (str): The location or label of the data segment being plotted.
    - window (int): The size of the window, i.e., the number of data points to be plotted.
    - index (int): The starting index of the window in the predictions and scores arrays.
    - predictions (list or array): The predicted values.
    - scores (list or array): The actual scores (ground truth).
    - data_name (str): The name of the dataset, used to organize the output file.
    - method_name (str): The name of the online conformal method, used in the plot label and output file name.
    """

    plt.clf()
    plt.plot(range(index, index + window), predictions[index : index + window], label=f"Predictions ({METHOD_NAME_LEGEND_MAP[method_name]})")
    plt.plot(range(index, index + window), scores[index : index + window], label="Scores")
    plt.legend(fontsize = 16)
    plt.title(f"{location} {window} scores", fontsize = 15)
    plt.xticks(fontsize=15)  # Set x-tick labels font size
    plt.yticks(fontsize=15)  # Set y-tick labels font size
    plt.tight_layout()
    plt.savefig(f'./{experiment_name}/plots/{data_name}/pred_vs_score_window_{location}/{method_name}.pdf', dpi=300, format='pdf')


def plot_distance_to_qt(location, window, index, qt_predictions, baseline_predictions, data_name, method_name, experiment_name):
    """
    Plots the difference between the predictions from linear quantile tracking and a baseline.

    Parameters:
    - location (str): The location or label of the data segment being plotted.
    - window (int): The size of the window, i.e., the number of data points to be plotted.
    - index (int): The starting index of the window in the predictions and scores arrays.
    - predictions (list or array): The predicted values.
    - scores (list or array): The actual scores (ground truth).
    - data_name (str): The name of the dataset, used to organize the output file.
    - method_name (str): The name of the baseline method.
    """

    sliced_qt_predictions = np.array(qt_predictions[index : index + window])
    sliced_baseline_predictions = np.array(baseline_predictions[index : index + window])
    differences = sliced_qt_predictions - sliced_baseline_predictions

    plt.clf()
    plt.plot(range(index, index + window), differences, label=f"linear_qt_fixed - {METHOD_NAME_LEGEND_MAP[method_name]}")
    plt.title(f"{location} {window} predictions")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=np.mean(differences), color='green', linestyle='--', label = 'Average on interval')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./{experiment_name}/plots/{data_name}/distance_to_qt_{location}/{method_name}.pdf', dpi=300)


def eval(data_name, experiment_name, scores=None, write_latex=False):
    """
    Creates all the plots given a dataset name whose results exist in ./results/

    Parameters:
    - data_name (str): The name of the dataset.
    """

    latex_rows = []

    # Read results from the JSON file
    with open(f'./{experiment_name}/results/{data_name}.json', 'r') as json_file:
        methods = json.load(json_file)

    # Get scores for plotting
    scores = get_scores(data_name, scores=scores)
    print(f'Plotting results for {data_name}')

    for method_name, method_details in methods.items():
        val_size = method_details['val_size']
        break
    alpha=0.1
        
    test_scores = scores[val_size:] 
    # Update if no validation set was used
    if len(method_details['predictions']) != len(test_scores):
        test_scores = scores
    T_test = len(test_scores)

    # Create directories for plots such as rolling_coverage if they don't already exist
    os.makedirs(f'./{experiment_name}/plots', exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/rolling_coverage/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/rolling_win_rate/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/rolling_win_rate/set_size'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/rolling_win_rate/quantile_loss'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/rolling_win_rate/absolute_loss'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/pred_vs_score_window_Middle/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/pred_vs_score_window_First/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/pred_vs_score_window_Final/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/distance_to_qt_First/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/distance_to_qt_Middle/'
    os.makedirs(directory, exist_ok=True)
    directory = f'./{experiment_name}/plots/{data_name}/distance_to_qt_Final/'
    os.makedirs(directory, exist_ok=True)


    # Plot bar charts comparing methods.
    for metric in METRICS:
        print(f'Plotting {metric}...')
        categories = methods.keys()
        values = [methods[method]['results'][metric] for method in categories]
        categories = [f'{METHOD_NAME_LEGEND_MAP[category]} (p: {methods[category]["kwargs"]["p_order"]}, b:{methods[category]["kwargs"]["bias"]})'  if 'bias' in methods[category]["kwargs"].keys() else METHOD_NAME_LEGEND_MAP[category] for category in categories]
        # Create a bar chart
        plt.clf()
        # Create a list for bar colors
        colors = ['orange' if 'batch' in category.lower() else 'blue' for category in categories]

        # Create the bar chart
        plt.bar(categories, values, color=colors)
        # Add the values above each bar
        # Get the variance in the predictions.
        for i, value in enumerate(values):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(metric)
        plt.xlabel('Method')
        plt.ylabel(metric)
        if metric == 'coverage':
            plt.axhline(y=1-alpha, color='r', linestyle='--')  # Red dashed line at y = 1 - alpha
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/{metric}.pdf', dpi=300) 

        if metric in ['coverage', 'quantile_loss', 'set_size']:
            latex_rows.append(values)


    # Get plots that involve computation between separate methods' predictions: win rate, full differences, and p vals.
    compare_method_name = 'linear_qt_fixed'
    compare_method_exists = compare_method_name in methods.keys()
    if compare_method_exists:
        categories = methods.keys() 
        comparison_predictions = np.array(methods[compare_method_name]['predictions'])
        comparison_quantile_losses = np.array([quantile_loss(comparison_predictions[i], test_scores[i], 1-alpha) for i in range(T_test)])
        print(np.std(comparison_quantile_losses) / len(test_scores))
        print(np.std(comparison_predictions)  / len(test_scores))
        comparison_absolute_losses = np.array([absolute_loss(comparison_predictions[i], test_scores[i]) for i in range(T_test)])
        win_rates = []
        win_rates_quantile_losses = []
        win_rates_absolute_losses = []
        difference_sequences = []
        mean_differences = []
        std_differences = []
        quantile_losses_difference_sequences = []
        absolute_losses_difference_sequences = []
        for method in categories:
            differences = comparison_predictions - np.array(methods[method]['predictions'])
            difference_sequences.append(differences)
            quantile_losses = [quantile_loss(methods[method]['predictions'][i], test_scores[i], 1-alpha) for i in range(T_test)]
            quantile_losses_difference_sequences.append(np.array(quantile_losses) - comparison_quantile_losses)
            absolute_losses = [absolute_loss(methods[method]['predictions'][i], test_scores[i]) for i in range(T_test)]
            absolute_losses_difference_sequences.append(np.array(absolute_losses) - comparison_absolute_losses)
            win_rates.append(np.mean((differences >= 0).astype(int)))
            win_rates_quantile_losses.append(np.mean((comparison_quantile_losses >= quantile_losses).astype(int)))
            win_rates_absolute_losses.append(np.mean((comparison_absolute_losses >= absolute_losses).astype(int)))
            mean_differences.append(np.mean(differences)) 
            std_differences.append(np.std(differences)) 
            
        # Create win rate plot for set size
        categories = [f'{category} (p: {methods[category]["kwargs"]["p_order"]}, b:{methods[category]["kwargs"]["bias"]})'  if 'bias' in methods[category]["kwargs"].keys() else category for category in categories]    
        plt.clf()
        plt.bar(categories, win_rates, color=colors)
        # Add the values above each bar
        for i, value in enumerate(win_rates):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'Win rate vs {compare_method_name}')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/win_rate_vs_qt.pdf', dpi=300) 

        # Create win rate plot for quantile loss
        plt.clf()
        plt.bar(categories, win_rates_quantile_losses, color=colors)
        # Add the values above each bar
        for i, value in enumerate(win_rates_quantile_losses):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'Win rate on quantile loss vs {compare_method_name}')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/win_rate_quantile_loss_vs_qt.pdf', dpi=300) 

        # Create win rate plot for absolute loss
        plt.clf()
        plt.bar(categories, win_rates_absolute_losses, color=colors)
        # Add the values above each bar
        for i, value in enumerate(win_rates_absolute_losses):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'Win rate on absolute loss vs {compare_method_name}')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/win_rate_absolute_loss_vs_qt.pdf', dpi=300) 
        latex_rows.append(win_rates_quantile_losses)
        latex_rows.append(win_rates)

        # Create differences plot
        plt.clf()
        plt.bar(categories, mean_differences, yerr=std_differences, color=colors)
        # Add the values above each bar
        for i, value in enumerate(mean_differences):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'Mean/std differences vs {compare_method_name}')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/distance_to_qt.pdf', dpi=300) 

        # Create p val plot for quantile loss
        plt.clf()
        vals = [one_sided_h_test(quantile_losses_difference_sequence)[1] for quantile_losses_difference_sequence in quantile_losses_difference_sequences]
        plt.bar(categories, vals, color=colors)
        # Add the values above each bar
        for i, value in enumerate(vals):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'p-val for quantile loss vs {compare_method_name}')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/p_val_quantile_loss.pdf', dpi=300) 
        latex_rows.append(vals)

        # Create p val plot for set size
        plt.clf()
        vals = [one_sided_h_test(-difference_sequence)[1] for difference_sequence in difference_sequences]
        plt.bar(categories, vals, color=colors)
        # Add the values above each bar
        for i, value in enumerate(vals):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'p-val for set size vs {compare_method_name}')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/p_val_set_size.pdf', dpi=300) 
        latex_rows.append(vals)

        # Create p val plot for absolute loss
        plt.clf()
        vals = [one_sided_h_test(absolute_losses_difference_sequence)[1] for absolute_losses_difference_sequence in absolute_losses_difference_sequences]
        plt.bar(categories, vals, color=colors)
        # Add the values above each bar
        for i, value in enumerate(vals):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'p-val for absolute loss vs {compare_method_name}')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/p_val_absolute_loss.pdf', dpi=300) 
        latex_rows.append(vals)

        # P val for win rate with quantile loss. 
        plt.clf()
        vals = [one_sided_h_test((quantile_losses_difference_sequence > 0).astype(int) - 0.5)[1] for quantile_losses_difference_sequence in quantile_losses_difference_sequences]
        plt.bar(categories, vals, color=colors)
        # Add the values above each bar
        for i, value in enumerate(vals):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'p-val for win rate on quantile loss vs {compare_method_name}')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/p_val_win_rate_quantile_loss.pdf', dpi=300) 
        latex_rows.append(vals)


        # P val for win rate with set size. 
        plt.clf()
        vals = [one_sided_h_test((difference_sequence < 0).astype(int) - 0.5)[1] for difference_sequence in difference_sequences]
        plt.bar(categories, vals, color=colors)
        # Add the values above each bar
        for i, value in enumerate(vals):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'p-val for win rate on set size vs {compare_method_name}')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/p_val_win_rate_set_size.pdf', dpi=300) 
        latex_rows.append(vals)

        # P val for win rate with absolute loss. 
        plt.clf()
        vals = [one_sided_h_test((absolute_losses_difference_sequence > 0).astype(int) - 0.5)[1] for absolute_losses_difference_sequence in absolute_losses_difference_sequences]
        plt.bar(categories, vals, color=colors)
        # Add the values above each bar
        for i, value in enumerate(vals):
            plt.text(i, value + 0.1, f'{value:.3f}', ha='center', fontsize=8, rotation=33)  # Rounded to 3 decimal places, smaller text
        plt.xticks(rotation=36, ha='right')
        # Add title and labels
        plt.title(f'p-val for win rate on absolute loss vs {compare_method_name}')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.xlabel('Method')
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/p_val_win_rate_absolute_loss.pdf', dpi=300) 
        latex_rows.append(vals)


    # Create individual plots for each method: rolling coverage and scores vs. predictions.
    for idx, (method_name, method_details) in enumerate(methods.items()):
        print(f'Running {method_name}...')
        # Create rolling coverage plots.
        plt.clf()
        W = 1000
        plt.plot(method_details['results'][f'{W}-smoothed coverages'][W:-W], label=method_name)
        plt.legend()
        plt.tight_layout()
        # Show the plot
        plt.savefig(f'./{experiment_name}/plots/{data_name}/rolling_coverage/{method_name}.pdf', dpi=300) 


        if compare_method_exists:
            W = 500
            # Create rolling_win_rate/set_size.
            plt.clf()
            plt.plot(smooth_array((difference_sequences[idx] >= 0).astype(int), W), label=method_name)
            plt.axhline(y=0.5, color='r', linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.title('Rolling win rate on set size vs linear_qt_fixed')
            # Show the plot
            plt.savefig(f'./{experiment_name}/plots/{data_name}/rolling_win_rate/set_size/{method_name}.pdf', dpi=300) 

            # Create rolling_win_rate/quantile_loss.
            plt.clf()
            plt.plot(smooth_array((quantile_losses_difference_sequences[idx] <= 0).astype(int), W), label=method_name)
            plt.axhline(y=0.5, color='r', linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.title('Rolling win rate on quantile_loss vs linear_qt_fixed')
            # Show the plot
            plt.savefig(f'./{experiment_name}/plots/{data_name}/rolling_win_rate/quantile_loss/{method_name}.pdf', dpi=300) 

            # Create rolling_win_rate/absolute_loss.
            plt.clf()
            plt.plot(smooth_array((absolute_losses_difference_sequences[idx] <= 0).astype(int), W), label=method_name)
            plt.axhline(y=0.5, color='r', linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.title('Rolling win rate on absolute_loss vs linear_qt_fixed')
            # Show the plot
            plt.savefig(f'./{experiment_name}/plots/{data_name}/rolling_win_rate/absolute_loss/{method_name}.pdf', dpi=300) 


        # Create scores vs prediction plot for middle.
        window = 300
        index = T_test//2
        location = 'Middle'
        plot_pred_vs_score_window(location, window, index, method_details['predictions'], test_scores, data_name, method_name, experiment_name)
        if compare_method_name in methods.keys():
            plot_distance_to_qt(location, window, index, methods[compare_method_name]['predictions'], method_details['predictions'], data_name, method_name, experiment_name)
       
        # Same for beginning and end.
        index = 0
        location = 'First'
        plot_pred_vs_score_window(location, window, index, method_details['predictions'], test_scores, data_name, method_name, experiment_name)
        if compare_method_name in methods.keys():
            plot_distance_to_qt(location, window, index, methods[compare_method_name]['predictions'], method_details['predictions'], data_name, method_name, experiment_name)

        index = T_test - window
        location = 'Final'
        plot_pred_vs_score_window(location, window, index, method_details['predictions'], test_scores, data_name, method_name, experiment_name)
        if compare_method_name in methods.keys():
            plot_distance_to_qt(location, window, index, methods[compare_method_name]['predictions'], method_details['predictions'], data_name, method_name, experiment_name)

    if write_latex and compare_method_exists:
        cov_constraint = 1 - alpha - 0.03
        # Write in a latex_out.txt file. 
        latex_rows = [[round(item, 3) for item in sublist] for sublist in latex_rows]
        latex_string = ""
        
        def get_table_result(i, method_indicies):
            # filter out methods that did not get coverage
            filtered_method_indicies = [index for index in method_indicies if latex_rows[0][index] >= cov_constraint]
            if len(filtered_method_indicies) == 0:
                filtered_method_indicies = method_indicies
            # return just the index
            candidates = [latex_rows[i][j] for j in filtered_method_indicies]
            val = max(candidates) if  i in [3, 4, 5, 6, 7, 8, 9, 10] else min(candidates)
            return filtered_method_indicies[candidates.index(val)]
        
        def make_bold_latex(text):
            return rf"\textbf{{{text}}}"

        def make_underline_latex(text):
            return rf"\underline{{{text}}}"
        
        def make_strikethrough_latex(text):
            return f"{text}$^*$" # rf"\st{{{text}}}"

        include_p_vals = len(test_scores) > 4000

        for i in range(1, len(latex_rows) if include_p_vals else len(latex_rows) - 6):
            index_groups = [[4], [10], [-1, -2], [2, 3], [0, 1], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
            selected_methods = [  get_table_result(i, indicies) for indicies in index_groups   ]
            method_values = [latex_rows[i][j] for j in selected_methods]

            bold_indicies = []
            #underlined_indicies = []
            strikethrough_indicies = [] 

            if i in [1, 2]:
                # Get bold numbers
                for j in range(len(method_values)):
                    if j < 3:
                        # Compare to baselines
                        if method_values[j] < 0.95 * min(method_values[3:]):
                            bold_indicies.append(j)
                    else:
                        # Compare to ours
                        if method_values[j] < 0.95 * min(method_values[:3]):
                            bold_indicies.append(j)

                    if latex_rows[0][selected_methods[j]] < cov_constraint:
                        strikethrough_indicies.append(j)

                # Get winner
                #underlined_indicies.append(method_values.index(min(method_values)))

            # Apply 
            formatted_strings = [str(result) for result in method_values]
            formatted_strings = [make_strikethrough_latex(string) if j in strikethrough_indicies else string for (j, string) in enumerate(formatted_strings)]
            formatted_strings = [make_bold_latex(string) if j in bold_indicies else string for (j, string) in enumerate(formatted_strings)]
            #formatted_strings = [make_underline_latex(string) if j in underlined_indicies else string for (j, string) in enumerate(formatted_strings)]

            print(bold_indicies)

            # Concat with &
            descriptions = ["q. loss (avg)" , "set size (avg)" , "q. loss (win \%)", "set size (win \%)", "q. loss (avg) p-val", "set size (avg) p-val", "a. loss (avg) p-val", "q. loss (win \%) p-val", "set size (win \%) p-val", "a. loss (win \%) p-val"]
            latex_string += f"{make_bold_latex(' '.join(data_name.split('_')[0:2])) if data_name!='elec' else make_bold_latex(data_name)}, {descriptions[i-1]} & " + " & ".join(formatted_strings)
            latex_string += '\\\\\n'

        latex_string += "\\midrule\n"       

        # Append the string to latex_out.txt
        with open(f"{experiment_name}/latex_out.txt", "a") as file:
            file.write(latex_string)



if __name__ == "__main__":
    # Test eval function.
    experiment_name = "pacf_p_order_val5"
    eval('elec', experiment_name)

    # for data_abbr in ['daily-climate']: # ['MSFT', 'AMZN', 'GOOGL', 'daily-climate']:
    #     for model_type in ['transformer']: #['theta', 'transformer', 'prophet', 'ar']:
    #         data_name = f'{data_abbr}_{model_type}_absolute-residual_scores' 
    #         eval(data_name, experiment_name)
