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
#base_dir = r"C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\used_models_ordered"
base_dir = r"E:\Documents\GIT\brainspy-tasks\tmp\input\models\ordered"
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
df_pert = pd.DataFrame(index=df_index, columns=['pert_abs','pert_abs_errors','pert_rel', 'pert_rel_errors'])
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
                df_filter = (input_elec, input_interval, model)
                rmse, errors = pert.get_perturbed_rmse(configs, compare_to_measurement=False, return_error=True)
                df_pert.loc[df_filter, method], df_pert.loc[df_filter, method+'_errors'] = rmse.item(), errors.flatten()

#%% Save data
print('Manually check if short description is correct!')
#print('save data manually!')
save_loc = r'E:\Documents\Afstuderen\results\pert_data_large.pkl'
df_pert.to_pickle(save_loc)