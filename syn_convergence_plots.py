import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
from tqdm import tqdm

import numpy as np
from methods import *
from load_data import get_scores


if __name__ == "__main__":
    experiment_name = "final"

    theta_star = np.array([-0.3, 0.3, 1.28881])
    fixed_lr_norms = []
    decaying_lr_norms = []

    NUM_RUNS = 10

    for i in range(NUM_RUNS):
        data_name = f'synthetic_AR_2_1M_{i}'

        with open(f'./{experiment_name}/results/{data_name}.json', 'r') as json_file:
            methods = json.load(json_file)


        offset = 1

        num_thetas_to_plot = 100
        # Generate x-values corresponding to the length of y_vals
        x_vals = np.linspace(1, 1_000_000, num_thetas_to_plot - offset)  # Evenly spaced between 1 and 1M


        # Get results from dataset
        for method_name, method_details in methods.items():
            if 'thetas' in method_details and not '2' in method_name and not '4' in method_name:
                num_thetas = len(method_details['thetas'])
                selected_thetas = []
                # Get 100 thetas
                
                for i in range(num_thetas_to_plot):
                    selected_thetas.append(method_details['thetas'][int((i / num_thetas_to_plot) * num_thetas)])

                if 'fixed' in method_name:
                    fixed_lr_norms.append([np.linalg.norm(theta - theta_star) for theta in selected_thetas][offset:])
                else:
                    decaying_lr_norms.append([np.linalg.norm(theta - theta_star) for theta in selected_thetas][offset:])


    fixed_means = [np.mean([ fixed_lr_norms[j][i]  for j in range(NUM_RUNS)]) for i in range(len(fixed_lr_norms[0]))]
    fixed_stds = [np.std([ fixed_lr_norms[j][i]  for j in range(NUM_RUNS)]) for i in range(len(fixed_lr_norms[0]))]

    decaying_means = [np.mean([ decaying_lr_norms[j][i]  for j in range(NUM_RUNS)]) for i in range(len(decaying_lr_norms[0]))]
    decaying_stds = [np.std([ decaying_lr_norms[j][i]  for j in range(NUM_RUNS)]) for i in range(len(decaying_lr_norms[0]))]
    
    plt.errorbar(x_vals, fixed_means, yerr = fixed_stds, label = 'Fixed learning rate')
    plt.errorbar(x_vals, decaying_means, yerr = decaying_stds, label = 'Decaying learning rate')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.legend(fontsize=15)

    #plt.title('Convergence on synthetic data (\\norm{\\theta - \\theta^*})')
    plt.savefig(f'./{experiment_name}/synth.pdf', dpi=300)
