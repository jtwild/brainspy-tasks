# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:47:22 2020

@author: Jochem
"""
# %% Import packages
from bspytasks.benchmarks.capacity.capacity_test import CapacityTest
from bspyalgo.utils.io import load_configs
import numpy as np
import matplotlib.pyplot as plt
# %% User data
base_configs = load_configs('configs/benchmark_tests/capacity/template_gd.json')
output_directory = r"C:/Users/Jochem/Desktop/2020_04_24_capacity_loop_hyperparameters_VC6/"
# Lists to loop over
max_attempts_list = [1, 5, 10, 30]
learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
loss_function_list = ['corrsig', 'bce']
nr_epochs_list = [100, 250, 750, 1250]
shape = [4,4,2,4]

# %% Loop over all defined lists
base_configs['capacity_test']['results_base_dir'] = output_directory
descrips = np.empty(shape, dtype=str)
capacities = np.full(shape, np.nan)
summaries = np.empty(shape, dtype=object)
for i in range(len(max_attempts_list)):
    for j in range(len(learning_rate_list)):
        for k in range(len(loss_function_list)):
            for l in range(len(nr_epochs_list)):
                configs = base_configs.copy()
                # Edit the configs
                configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['max_attempts'] = max_attempts_list[i]
                configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['hyperparameters']['learning_rate'] = learning_rate_list[j]
                configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['hyperparameters']['loss_function'] = loss_function_list[k]
                configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['hyperparameters']['nr_epochs'] = nr_epochs_list[l]

                # Start the test
                test = CapacityTest(configs['capacity_test'])
                test.run_test(validate=False)
                plt.close('all')
                
                # Save some extra results
                descrips[i,j,k,l] = f"max_attempts: {max_attempts_list[i]}, learning_rate: {learning_rate_list[j]}, loss_function: {loss_function_list[k]}, nr_epochs: {nr_epochs_list[l]}" 
                capacities[i,j,k,l] = test.summary_results['capacity_per_N'][0]
                summaries[i,j,k,l] = test.summary_results
np.savez(output_directory + 'loop_items.npz', capacities=capacities, summaries = summaries, descrips = descrips, 
         max_attempts_list = max_attempts_list, learning_rate_list = learning_rate_list, 
         loss_function_list = loss_function_list, nr_epochs_list = nr_epochs_list)
