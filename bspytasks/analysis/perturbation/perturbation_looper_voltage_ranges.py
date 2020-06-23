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
methods = ['pert_abs'] # pert_abs and pert_rel possible
num_ranges= 1000

#%% Create pandas dataframe to store results
df_index = pd.MultiIndex.from_product([input_elecs, input_intervals, models], names=['input_elec','input_interval','model'])
df_columns = pd.MultiIndex.from_product([methods,['score','batch_avgs','batch_ranges','batch_num_samples','batch_rmses']])
df_pert = pd.DataFrame(index=df_index, columns=df_columns)

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
                # Get per voltage range data
                batch_errors_subsets, batch_avgs, batch_ranges = pert.sort_by_input_voltage(inputs_unperturbed[:, input_elec], errors.flatten(), num_ranges = num_ranges)
                batch_rmses = [np.sqrt(np.mean(batch_errors**2)) for batch_errors in batch_errors_subsets]
                batch_num_samples = [batch_errors.size for batch_errors in batch_errors_subsets]

                # FIll the dataframe
                df_pert.loc[index_filter, (method, 'score')] = rmse.item()
                df_pert.loc[index_filter, (method, 'batch_avgs')] = batch_avgs
                df_pert.loc[index_filter, (method, 'batch_ranges')] = batch_ranges
                df_pert.loc[index_filter, (method, 'batch_num_samples')] = batch_num_samples
                df_pert.loc[index_filter, (method, 'batch_rmses')] = batch_rmses

#%% Save data
print('Manually check if short description is correct!')
#print('save data manually!')
save_loc = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\perturbation_results\2020_06_23_large_perturbation_run\perturbation_data.pkl'
df_pert.to_pickle(save_loc)
