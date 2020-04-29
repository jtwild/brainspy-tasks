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
import os
import glob
from bspyproc.processors.simulation.surrogate import SurrogateModel
# %% User data
base_configs = load_configs('configs/benchmark_tests/capacity/template_gd.json')
output_directory = r"C:/Users/Jochem/Desktop/2020_04_29_capacity_loop_7_models/"
# Lists to loop over
input_indices_list = [[0], [1], [2], [3], [4], [5], [6]]
#torch_model_dict_list taken from model list given
#voltage_intervals_list  differs per model, fixed in loop below

# %% Auto load torch models from directory
base_dir = r"E:\Documents\GIT\brainspy-tasks\tmp\input\models\ordered"
glob_filter = '**/*.pt'
glob_query = os.path.join(base_dir, glob_filter)
# And finally find the models we want to use
torch_model_dict_list = glob.glob(glob_query, recursive=True)
# Get min/max values for voltage ranges
voltage_intervals_list = np.zeros([len(input_indices_list), len(torch_model_dict_list), 2]) # last dimension is two to store low and high values
for model_index in range(len(torch_model_dict_list)):
    temp_processor = SurrogateModel({'torch_model_dict': torch_model_dict_list[model_index]})
    min_voltage = temp_processor.min_voltage.numpy()
    max_voltage = temp_processor.max_voltage.numpy()
    voltage_intervals_list[:,model_index,0] = min_voltage
    voltage_intervals_list[:, model_index, 1] = max_voltage
    # indices: first index is the electrode index, second index is the model index, third index contains min/max values
        
# %% Get shape
shape = [len(input_indices_list), len(torch_model_dict_list)]
# %% Loop over all defined lists
base_configs['capacity_test']['results_base_dir'] = output_directory
descrips = np.empty(shape, dtype=str)
capacities = np.full(shape, np.nan)
summaries = np.empty(shape, dtype=object)
for i in range(len(input_indices_list)):
    for j in range(len(torch_model_dict_list)):
        configs = base_configs.copy()
        # Edit the configs
        configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = input_indices_list[i]
        configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['torch_model_dict'] = torch_model_dict_list[j]
        configs['capacity_test']['vc_dimension_test']['voltage_intervals'] = voltage_intervals_list[i,j,:] #do I need the tolist() method?

        # Start the test
        test = CapacityTest(configs['capacity_test'])
        test.run_test(validate=False)
        plt.close('all')
        
        # Save some extra results
        descrips[i,j] = f"input_index = {input_indices_list[i]}, voltage_intervals = {voltage_intervals_list[i,j,:]}, model = {torch_model_dict_list[j]}" 
        capacities[i,j] = test.summary_results['capacity_per_N'][0]
        summaries[i,j] = test.summary_results
# Save loop items        
np.savez(output_directory + 'loop_items.npz', capacities=capacities, summaries = summaries, descrips = descrips, 
         input_indices_list = input_indices_list, 
         torch_model_dict_list = torch_model_dict_list, 
         voltage_intervals_list = voltage_intervals_list,
         shape = shape,
         loop_order = ['input_indices_list','torch_model_dict_list'])
