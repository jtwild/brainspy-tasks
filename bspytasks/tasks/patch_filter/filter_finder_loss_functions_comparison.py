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
import sys
from io import StringIO as StringIO

# TO prevent printing
class NullIO(StringIO):
    def write(self, txt):
       pass

loss_fns = ['entropy_hard_boundaries']
inputs = [[0,3,4]]
batch_sizes = [4]
types = ['dnpu', 'IOnet','IOnet','IOnet','IOnet']
scaling = [None, 'single_scaler', 'single_scaler', 'multi_scaler','multi_scaler']
regularizers = [[None, None], [76, -150], [76, -320], [76, -150], [76, -320]] # not be looped seperately, but combine with the above


base_configs = load_configs('configs/tasks/filter_finder/template_ff_gd.yaml')
for batch in batch_sizes:
    for loss in loss_fns:
        for inp in inputs:
            for i in range(len(types)):
                # Change the configs
                configs = copy.deepcopy(base_configs )
                configs['filter_finder']['algorithm_configs']['hyperparameters']['loss_function'] =  loss
                configs['filter_finder']['algorithm_configs']['hyperparameters']['batch_size'] =  batch
                configs['filter_finder']['algorithm_configs']['processor']['input_indices'] = inp
                configs['filter_finder']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = inp

                configs['filter_finder']['algorithm_configs']['processor']['network_type'] = types[i]
                configs['filter_finder']['algorithm_configs']['processor']['IOinfo']['output_high']  = regularizers[i][0]
                configs['filter_finder']['algorithm_configs']['processor']['IOinfo']['output_low']  = regularizers[i][1]
                configs['filter_finder']['algorithm_configs']['processor']['IOinfo']['mode'] = scaling[i]

                # Run the task
                print('\n\n---------------------------------------------')
                print(f'     starting loss {loss} and inputs {inp}   ')
                print('---------------------------------------------\n\n')
                sys.stdout = NullIO()
                task = FilterFinder(configs['filter_finder'], is_main=True) #initialize class
                excel_results = task.find_filter()
                plt.close("all")
                sys.stdout = sys.__stdout__
