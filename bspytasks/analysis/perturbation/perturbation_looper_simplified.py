# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:10:22 2020

@author: Jochem

Goal: loop over different models and different electrodes to get the rmse/error distribution of all of them.
Only get numerical values, no plotting whatsoever.
"""
# Load packages
import numpy as np
import pandas as pd
import os
import glob
import bspytasks.analysis.perturbation.perturbation_utils as pert
from bspyalgo.utils.io import load_configs

# %% Config data
base_configs = load_configs('configs/analysis/perturbation/single_perturbation_all_electrodes_configs.json')

# %% Auto load torch models from given directory
base_dir = r"C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\used_models_ordered"
glob_filter = '**/*.pt'
glob_query = os.path.join(base_dir, glob_filter)
# And finally find the models we want to use
torch_model_dict_list = glob.glob(glob_query, recursive=True)

#%% Define lists of what we are going to loop over
models = ['brains1','darwin1','darwin2','brains2.1','brains2.2','pinky1','darwin3'] #hardcoded... check with torch model dict list
input_elecs = [0, 1, 2, 3, 4, 5, 6]
input_intervals= ['full']

#%% Create pandas dataframe to store results
df_index = pd.MultiIndex.from_product([input_elecs, input_intervals, models], names=['input_elec','input_interval','model'])
df_columns = pd.MultiIndex.from_product([['pert_abs', 'pert_rel'],['score','errors','inputs_unpert','inputs_pert','outputs_unpert','outputs_pert']])
df_pert = pd.DataFrame(index=df_index, columns=df_columns)
methods = ['pert_abs','pert_rel']

for i, input_elec in enumerate(input_elecs):
    for j, input_interval in enumerate(input_intervals):
        for k, model in enumerate(models):
            for method in methods:

                # Set configs
                configs = base_configs.copy()
                configs['processor']['torch_model_dict'] = torch_model_dict_list[k]
                configs['perturbation']['electrodes_sets'] = [[input_elec]]
                if method == 'pert_abs':
                    configs['perturbation']['mode'] = 'absolute'
                elif method == 'pert_rel':
                    configs['perturbation']['mode'] = 'relative'
                else:
                    raise ValueError('Unknown method supplied.')

                # Get data
                index_filter = (input_elec, input_interval, model)
                rmse, errors, inputs_unperturbed, inputs_perturbed, targets, prediction = pert.get_perturbed_rmse(configs, compare_to_measurement=False)
                df_pert.loc[index_filter, (method, 'score')] = rmse.item()
                df_pert.loc[index_filter, (method, 'errors')] = errors.flatten()
                df_pert.loc[index_filter, (method, 'inputs_unpert')] = inputs_unperturbed
                df_pert.loc[index_filter, (method, 'inputs_pert')] = inputs_perturbed
                df_pert.loc[index_filter, (method, 'outputs_unpert')] = targets
                df_pert.loc[index_filter, (method, 'outputs_pert')] = prediction
#%% Save data
print('Manually check if short description is correct!')
print('save data manually!')
