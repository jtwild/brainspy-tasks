# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:21:44 2020

@author: Jochem
"""
# %% Load packages
import numpy as np
import matplotlib.pyplot as plt
import bspytasks.validation.electrode_ranking.perturbation_utils as pert
from bspyalgo.utils.io import load_configs

# %% Load User variables
configs = load_configs('configs/validation/single_perturbation_all_electrodes_configs.json')

# Get config values
electrodes_sets = configs['perturbation']['electrodes_sets']
perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']
#%% Get unperturbed data
inputs_unperturbed, targets_loaded, info = pert.load_data(configs)
targets = pert.get_prediction(configs, inputs_unperturbed).flatten()
#%% Initialize values
rmse = np.zeros((len(perturb_fraction_sets), len(electrodes_sets)))
fig_hist, axs_hist = plt.subplots(2, 4)
axs_hist = axs_hist.flatten()
fig_bar, axs_bar = plt.subplots(2, 4)
axs_bar = axs_bar.flatten()
counter = 0
for i in range(len(perturb_fraction_sets)):
    configs['perturbation']['perturb_fraction'] = perturb_fraction_sets[i]
    for j in range(len(electrodes_sets)):
        configs['perturbation']['electrodes'] = electrodes_sets[j]
        electrode = electrodes_sets[j][0]  # does not work if more than one electrode is perturbed..., so 'fixed' by taking only first component.
        # Perturb data, get prediciton, get error, get rmse
        inputs_perturbed = pert.perturb_data(configs, inputs_unperturbed)
        prediction = pert.get_prediction(configs, inputs_perturbed)
        # Real error
        error = prediction - targets  # for unkown sizes can use lists [([[]]*10)]*5 and convert to numpy afterwards
        error_subsets, grid, ranges = pert.sort_by_input_voltage(inputs_unperturbed[:, electrode], error,
                                                                 min_val=-1.2, max_val=0.6, granularity=0.1)
        pert.plot_hists(np.abs(error_subsets), ax=axs_hist[counter], legend=grid.round(2).tolist())
        pert.rank_low_to_high(grid,
                              np.sqrt(pert.np_object_array_mean(error_subsets**2)),
                              plot_type='ranking', ax=axs_bar[counter], x_data = grid)
        axs_bar[counter].set_xlabel('Voltage (V)')
        # And root mean square error
        rmse[i, j] = np.sqrt(np.mean(error**2))
        counter += 1

#%% Visualize results
fig_bar.suptitle('Voltage range ranking based on RMSE on interval')
# Ranking electrode importance
pert.rank_low_to_high(electrodes_sets, rmse[0, :], plot_type='ranking')
plt.title('Electrode ranking based on RMSE')
