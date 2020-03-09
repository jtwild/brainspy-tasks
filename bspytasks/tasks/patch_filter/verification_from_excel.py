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
import torch
import pandas as pd
from ast import literal_eval
from bspyproc.bspyproc import get_processor
from bspytasks.tasks.patch_filter.validation import PatchFilterValidator
#%% Define the timestamps that you want to check, and the models
timestamps = ['2020_03_09_21h15m35s', '2020_03_09_21h15m34s', '2020_03_09_21h15m33s', '2020_03_09_21h15m32s']
base_folder = r"C:\Users\Jochem\STACK\Daily_Usage\GIT_SourceTree\brainspy-tasks\tmp\output\filter_finder\march_single\useless_test"
file = r"filter_finder_collective_results.xlsx"
proc_base_folder = r'C:\Users\Jochem\STACK\Daily_Usage\GIT_SourceTree\brainspy-tasks\tmp\output\filter_finder\march_single\useless_test\patch_filter_4_points_2020_03_09_201537.35\gradient_descent_data\gradient_descent_data_2020_03_09_201537.39\reproducibility'
proc_file = r'results.pickle'
verification_configs_path = r'configs\tasks\filter_finder\template_gd_cdaq_to_nidaq_validation.json'

#%% Loading the data from the excel file:
file_path = os.path.join(base_folder, file)
df_full = pd.read_excel(file_path, engine = 'openpyxl')

df_selection = pd.DataFrame(columns = df_full.columns)
for ts in timestamps:
    df_selection = df_selection.append(df_full[df_full['timestamp']==ts])
df_selection = df_selection.reset_index(drop=True)

#%% And loading the (a) processor
proc_file_path = os.path.join(proc_base_folder, proc_file)
with open(proc_file_path, 'rb') as input_file:
    full_results = pickle.load(input_file)
    processor = full_results['processor']

#%% Calculating the new output again to see if it agrees with the saved outpyut.
df_selection["output_verification_simulation"] = ""
df_selection["output_difference_simulation"] = ""
for i in range(len(df_selection)):
    # Strip tensor(...) from strings if it is still there, and convert to list, and convert to tensor. Because of wrong old save format.
    controls = torch.tensor( literal_eval( df_selection.loc[i,'control_voltages'].strip('tensor(').strip(')') ) )
    inputs = torch.tensor( literal_eval(df_selection.loc[i,'input points'] ) )
    processor.bias.data[:] = controls
    if df_selection.loc[i,'scaling'] == ('multi_scaling' or 'single_scaling'):
        output_verification_simulation = processor.forward_without_scaling(inputs).detach().numpy().tolist()
    else:
        output_verification_simulation = processor.forward(inputs).detach().numpy().tolist()
    df_selection.loc[i,'output_verification_simulation'] = output_verification_simulation
    diff = np.array(output_verification_simulation) - np.array(literal_eval(df_selection.loc[i,'output points']))
    df_selection.loc[i, 'output_difference_simulation'] = diff
    print(diff)

# #%% Get output from hardware processor...
# verification_configs = load_configs(verification_configs_path)
# model_path = os.path.join(proc_base_folder, 'model.pt')
# model = torch.load('model.pth')
# val = PatchFilterValidator(configs)


# get_output(...) function, inputs numpy and outputs numpy. From the validation processor.