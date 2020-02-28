# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:13:57 2020

@author: Jochem
"""

from bspyalgo.utils.io import load_configs
from bspytasks.tasks.patch_filter.filter_finder import FilterFinder
from bspytasks.benchmarks.vcdim.vc_dimension_test import VCDimensionTest
import copy

loss_fns = ['sigmoid_distance', 'entropy', 'sigmoid_distance']
inputs = [[1,2],[0,3,4],[1,2,3,4]]
base_configs = load_configs('configs/tasks/filter_finder/template_ff_gd.json')
for loss in loss_fns:
    for inp in inputs:
        # Change the configs
        configs = copy.deepcopy(base_configs )
        configs['filter_finder']['algorithm_configs']['hyperparameters']['loss_function'] =  loss
        configs['filter_finder']['algorithm_configs']['processor']['input_indices'] = inp
        configs['filter_finder']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = inp
        # Run the task
        print('\n\n---------------------------------------------')
        print(f'     starting loss {loss} and inputs {inp}   ')
        print('---------------------------------------------\n\n')
        task = FilterFinder(configs['filter_finder'], is_main=True) #initialize class
        excel_results = task.find_filter()
