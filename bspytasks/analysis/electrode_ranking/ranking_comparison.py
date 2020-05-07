# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:31:28 2020

@author: Jochem

Goal:
    - load data from
        - pertrubation ranking
        - gradient ranking
        - VC dimension
    for different models, and for different electrodes
and compare them

"""

#%% Importing packages
import numpy as np
import matplotlib.pyplot as plt
import bspytasks.analysis.electrode_ranking.ranking_utils as rank_utils
import os
#%% Defining user variables locations
n_elec = 7 # Hardcoded because all data must match this
n_intervals = 1
n_models = 7
shape = [n_elec, n_intervals, n_models]
# Looped in order: [input indices, voltage_intervals, torch_model_dict]

# Load npz libraires containing the data. Contains a lot more information for checking data. Not used in this script, feel free to explore.
gradient_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\gradient_results\loop_items_gradient.npz')
perturbation_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_05_01_perturbation_results\perturbation_results.npz')
vc_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_04_29_capacity_loop_7_models\loop_items.npz', allow_pickle = True)
models = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\model_description.npz')['model_description']

#%% Load data from npz libraries
gradient = gradient_lib['gradient']
perturbation = perturbation_lib['rmse']

# Get VC data from summaries, because it was not saved correctly:
vc_summaries = vc_lib['summaries']
n_vcs = 5
vc = np.full(np.append(shape, n_vcs), np.nan)
for i in range(n_elec):
    for j in range(n_models):
        #zeroth dimension hardcoded, because I did not loop over the one voltage intervals in this case
        vc[i,0,j, :] = vc_summaries[i,j]['capacity_per_N']
vc6 = vc[:,:,:,4]

#%% Plot different dfigures
print('Check manually if inputs are formatted according ot the same standard and that the same models/eelctrodes are used!')

# comparing methods and ranking: 7 subplots (one per device). y-axis rank, x-axis electrodes, bars for methods (so 3 bars per electrodes)
fig_a, ax_a  = plt.subplots(nrows=4, ncols =1, sharey=True)


