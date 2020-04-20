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
output_directory = r"C:/Users/Jochem/STACK/Daily_Usage/Bestanden/UT/TN_MSc/Afstuderen/Results/Electrode_importance/2020_04_19_capacity_loop/"
# Lists to loop over
num_loops = 3
voltage_intervals_list = [[-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3]]
input_indices_list = [[0], [4], [6]]

# %% Loop over all defined lists
base_configs['capacity_test']['results_base_dir'] = output_directory
for i in range(num_loops):
    configs = base_configs.copy()
    configs['capacity_test']['vc_dimension_test']['voltage_intervals'] = voltage_intervals_list[i]
    configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = input_indices_list[i]

    test = CapacityTest(configs['capacity_test'])
    test.run_test(validate=False)
    plt.close('all')
np.savez(output_directory + 'loop_items.npz', voltage_intervals_list=voltage_intervals_list, input_indices_list=input_indices_list)
