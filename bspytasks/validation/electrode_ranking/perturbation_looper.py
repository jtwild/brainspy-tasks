# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:27:11 2020

@author: Jochem
"""
#%% Loading packages
from bspytasks.validation.electrode_ranking.perturbation_electrode_ranking import ElectrodeRanker
from bspyalgo.utils.io import load_configs
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# %% User data
base_configs = load_configs('configs/validation/single_perturbation_all_electrodes_configs.json')
output_directory = r"C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\perturbation_results"
# Lists to loop over
#input_indices_list = [[0], [1], [2], [3], [4], [5], [6]] -> this is already contained in the perturbation looper
#torch_model_dict_list taken from model list given
#voltage_intervals_list  differs per model, fixed in loop below

# %% Auto load torch models from given directory
base_dir = r"C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\used_models_ordered"
glob_filter = '**/*.pt'
glob_query = os.path.join(base_dir, glob_filter)
# And finally find the models we want to use
torch_model_dict_list = glob.glob(glob_query, recursive=True)

# %% Get shape
input_indices_len = 7
voltage_interval_len = 1
shape = [input_indices_len, voltage_interval_len, len(torch_model_dict_list)]

# %% Loop over all defined lists
descrips = np.empty(shape, dtype=object)
rmse = np.full(shape, np.nan)
ranked_descriptions = np.full(shape, np.nan)
for j, torch_model_dict in enumerate(torch_model_dict_list):
    configs = base_configs.copy()
    # Edit the configs
    configs['processor']['torch_model_dict'] = torch_model_dict

    # Start the test
    ranker = ElectrodeRanker(configs)
    rmse[:,:,j] = ranker.rank().T
    ranked_descriptions[:,:,j], ranked_values, ranking_indices = ranker.plot_rank()
#    plt.close('all')

    # Save some extra results
    descrips[:,:,j] = f"model = {torch_model_dict_list[j]}"
# Save loop items
np.savez(output_directory + 'loop_items.npz', rmse=rmse, ranked_descriptions = ranked_descriptions,
         descrips = descrips,
         torch_model_dict_list = torch_model_dict_list,
         shape = shape,
         loop_order = ['input_indices', 'voltage_intervals', 'torch_model_dict_list'])
