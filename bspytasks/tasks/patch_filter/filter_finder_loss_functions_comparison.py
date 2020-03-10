# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:13:57 2020

@author: Jochem
"""

from bspyalgo.utils.io import load_configs
from bspytasks.tasks.patch_filter.filter_finder import FilterFinder
from bspytasks.benchmarks.vcdim.vc_dimension_test import VCDimensionTest
import matplotlib.pyplot as plt
import copy

loss_fns = ['sigmoid_nn_distance']*20
batch_sizes = [4]
learning_rates = [0.0005]
input_sets = [[[0, 1]]]
# combined loops:
inputs = [[1,2,3,4]]
types = ['IOnet'] #combined loop
scaling = ['multi_scaler'] #combined loop
#regularizers = [[110,-110]]             # not be looped seperately, but combine with the above


base_configs = load_configs('configs/tasks/filter_finder/template_ff_gd.yaml')
loop=0
max_loop = len(loss_fns) * len(batch_sizes) * len(learning_rates) * len(types) * len(input_sets)
for lr in learning_rates:
    for sets in input_sets:
        for i in range(len(types)):
            for batch in batch_sizes:
                for loss in loss_fns:
                    loop+=1
                    # Change the configs
                    configs = copy.deepcopy(base_configs )
                    configs['filter_finder']['algorithm_configs']['hyperparameters']['loss_function'] =  loss
                    configs['filter_finder']['algorithm_configs']['hyperparameters']['batch_size'] =  batch
                    configs['filter_finder']['algorithm_configs']['hyperparameters']['learning_rate'] = lr
                    configs['filter_finder']['boolean_gate_test']['algorithm_configs']['processor']['point_generation_sets'] = sets
                    #Combined loop:
                    configs['filter_finder']['algorithm_configs']['processor']['input_indices'] = inputs[i]
                    configs['filter_finder']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = inputs[i]
                    configs['filter_finder']['algorithm_configs']['processor']['network_type'] = types[i]
                    # configs['filter_finder']['algorithm_configs']['processor']['IOinfo']['output_high']  = regularizers[i][0]
                    # configs['filter_finder']['algorithm_configs']['processor']['IOinfo']['output_low']  = regularizers[i][1]
                    configs['filter_finder']['algorithm_configs']['processor']['IOinfo']['mode'] = scaling[i]

                    # Run the task
                    print('\n\n---------------------------------------------')
                    print(f'     starting loop {loop} out  {max_loop}   ')
                    print('---------------------------------------------\n\n')
                    task = FilterFinder(configs['filter_finder'], is_main=True) #initialize class
                    excel_results = task.find_filter()
                    plt.close("all")
