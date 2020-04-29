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
output_directory = r"C:/Users/Jochem/Desktop/2020_04_21_capacity_loop_all_electrodes_different_models/"
# Lists to loop over
# The first loop loops over the models
num_loops_1 = 3
model_list = [[r'tmp\input\models\ordered\chip1\Darwin\measured2020_03_10\modelled_2020_03_11\model.pt'],
              [r'tmp\input\models\ordered\chip2\Brains\measured2020_03_11\modelled2020_03_19\model.pt'],
              [r'tmp\input\models\ordered\chip3\Pinky\measured2020_04_05\modelled_2020_03_11\model.pt']]
# The second loop loops over the electrodes
num_loops_2 = 7
voltage_intervals_list = [[-1.2, 0.6], [-1.2, 0.6],[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3],[-0.7, 0.3]]
input_indices_list = [[0], [1],[2],[3],[4],[5], [6]]

# %% Loop over all defined lists
base_configs['capacity_test']['results_base_dir'] = output_directory
descrip = list()
for i in range(num_loops_1):
    configs = base_configs.copy()
    # Update configs for this loop
    configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['torch_model_dict'] = model_list[i]
    for j in range(num_loops_2):
        configs['capacity_test']['vc_dimension_test']['voltage_intervals'] = voltage_intervals_list[j]
        configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = input_indices_list[j]

        # Start the test
        descrip.append(model_list[i]+'_'+input_indices_list[j]+'_'+voltage_intervals_list[j])
        test = CapacityTest(configs['capacity_test'])
        test.run_test(validate=False)
        plt.close('all')
np.savez(output_directory + 'loop_items.npz', voltage_intervals_list=voltage_intervals_list, input_indices_list=input_indices_list, model_list=model_list, descrip=descrip)
