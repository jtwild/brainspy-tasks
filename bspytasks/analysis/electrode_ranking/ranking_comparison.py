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

# Manual information about the loops
descr_elec = [0, 1, 2, 3, 4, 5, 6]
descr_intervals = ['full']
descr_models = models
descr_methods = ['grad', 'pert', 'vc6']

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


#%% Rank all data:
rank_perturbation = np.full(shape, np.nan)
rank_gradient = np.full(shape, np.nan)
rank_vc6 = np.full(shape, np.nan)
for j in range(n_intervals):
    for k in range(n_models):
        rank_perturbation[:,j,k] = rank_utils.rank_low_to_high(perturbation[:,j,k])[1] # [1] selects the ranking indices. A.k.a. the rank of the different elements.
        rank_gradient[:,j,k] = rank_utils.rank_low_to_high(gradient[:,j,k])[1]
        rank_vc6[:,j,k] = rank_utils.rank_low_to_high(vc6[:,j,k])[1]
# Add one to set zero score to one.
rank_perturbation+=1
rank_gradient+=1
rank_vc6+=1
#%% Plot different dfigures
print('Check manually if inputs are formatted according ot the same standard and that the same models/eelctrodes are used!')

# comparing methods and ranking: 7 subplots (one per device). y-axis rank, x-axis electrodes, bars for methods (so 3 bars per electrodes)
fig_a, ax_a  = plt.subplots(nrows=2, ncols=4, sharey=True)
ax_a = ax_a.flatten()
for i in range(n_models):
    data = np.stack((rank_gradient[:,0,i],
                     rank_perturbation[:,0,i],
                     rank_vc6[:,0,i]),
                    axis=0)
    legend = descr_methods
    xticklabels = descr_elec
    ax = ax_a[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(descr_models[i])


