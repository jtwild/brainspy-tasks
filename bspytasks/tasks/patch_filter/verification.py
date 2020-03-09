# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:47:54 2020

@author: Jochem

Goal: verify functionality found by gradient descent algorithm.
TImestamps of interesting configs (all collected in best_outputs):
    4 points:
        Single scaling: 6 to 7 na+A
            2020_03_07_16h04m57s
            2020_03_08_17h30m33s
            2020_03_08_23h10m59s
            2020_03_08_18h33m12s
        Multi scaling: >8 nA
            2020_03_07_03h15m14s
            2020_03_06_23h30m14s
            2020_03_08_06h38m05s --> Min -250nA, relatively okay interval!
            2020_03_07_03h57m37s
            2020_03_06_22h40m46s
            2020_03_06_15h13m45s
    3 points:
        No scaling: ~20nA
            2020_03_08_03h42m44s
            2020_03_08_03h30m30s
            2020_03_08_04h07m20s
            2020_03_08_03h08m17s
            2020_03_08_03h54m59s
        Single scaling: ~20nA
            2020_03_07_22h09m49s
            2020_03_07_21h49m32s
            2020_03_07_23h00m12s
            2020_03_07_22h40m11s
            2020_03_07_21h39m22s
            2020_03_07_23h22m22s
            (also smaller interval available, but then max 10nA seperation)
"""

import pickle
import os
import numpy as np

# Loading data:
base_folder = r"C:\Users\Jochem\STACK\Daily_Usage\GIT_SourceTree\brainspy-tasks\tmp\output\filter_finder\model_march\single_attempts\run1\patch_filter_4_points_2020_03_09_130109\gradient_descent_data_2020_03_09_130109\reproducibility"
rel_file = r"results.pickle"
location = os.path.join(base_folder, rel_file)
with open(location, 'rb') as input_file:
    results = pickle.load(input_file)
inputs = np.array(results['processor'].input)
controls = np.array(results['processor'].get_control_voltages())

processor = results['processor']
import torch
inputs_torch = torch.tensor(inputs)
scaling = processor.scaling
offset = processor.offset

# Get output
new_outputs = processor.forward_without_scaling(inputs_torch).detach().numpy()
new_outputs.sort(axis=0)