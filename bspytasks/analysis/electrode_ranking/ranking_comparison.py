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

# %% Importing packages
import numpy as np
import matplotlib.pyplot as plt
import bspytasks.analysis.electrode_ranking.ranking_utils as rank_utils
import os
# %% Defining user variables locations
n_elec = 7  # Hardcoded because all data must match this
n_intervals = 1
n_models = 7
shape = [n_elec, n_intervals, n_models]
# Looped in order: [input indices, voltage_intervals, torch_model_dict]

# Load npz libraires containing the data. Contains a lot more information for checking data. Not used in this script, feel free to explore.
gradient_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_05_14_gradient_results\loop_items_gradient.npz')
perturbation_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_05_01_perturbation_results\perturbation_results.npz')
vc_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_05_14_capacity_loop_7_models_VC8\loop_items.npz', allow_pickle=True)
models = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\model_description.npz')['model_description']

# Manual information about the loops
descr_elec = np.array([0, 1, 2, 3, 4, 5, 6])
descr_intervals = ['full']
descr_models = models
descr_models_short = np.array(['brains1', 'darwin1', 'darwin2', 'brains2.1', 'brains2.2', 'pinky1','darwin3'])
descr_methods = np.array(['grad', 'pert', 'vcX'])

# Manual selection of specific model indices.
#selec_elec = [0,1,2,3,4,5,6] # no selection currently performed in electrodes
#selec_intervals= [0] # no selection currently performed on intervals
# select only specifc models, to investigate brains/darwin/pinky difference
#select_models = [0, 3, 4]  # brains models
#select_models = [1, 2, 6] # Darwin models
select_models = [5] # pinky models.

# %% Load data from npz libraries
gradient = gradient_lib['gradient']
perturbation = -perturbation_lib['rmse']
print('Warning! Perturbation loaded negatively such that large (positive) perturbations gets a high rank value (high rank = meaning unimportant)')

# Get VC data from summaries, because it was not saved correctly:
vc_summaries = vc_lib['summaries']
n_vcs = 1
vc = np.full(np.append(shape, n_vcs), np.nan)
for i in range(n_elec):
    for j in range(n_models):
        # zeroth dimension hardcoded, because I did not loop over the one voltage intervals in this case
        vc[i, 0, j, :] = vc_summaries[i, j]['capacity_per_N']
vcX = vc[:, :, :, -1]


# %% Rank all data:
rank_gradient = np.full(shape, np.nan)
rank_perturbation = np.full(shape, np.nan)
rank_vcX = np.full(shape, np.nan)
for j in range(n_intervals):
    for k in range(n_models):
        rank_gradient[:, j, k] = rank_utils.rank_low_to_high(gradient[:, j, k])[0]
        rank_perturbation[:, j, k] = rank_utils.rank_low_to_high(perturbation[:, j, k])[0]
        rank_vcX[:, j, k] = rank_utils.rank_low_to_high(vcX[:, j, k])[0]
#print('vcX rank inverted. High VC score = low rank')
# Add one to set zero score to one.
rank_gradient += 1
rank_perturbation += 1
rank_vcX += 1

# %% Normalize data
norm_perturbation = np.full(shape, np.nan)
norm_gradient = np.full(shape, np.nan)
norm_vcX = np.full(shape, np.nan)
for j in range(n_intervals):
    for k in range(n_models):
        norm_gradient[:, j, k] = rank_utils.normalize(gradient[:, j, k])
        norm_perturbation[:, j, k] = rank_utils.normalize(perturbation[:, j, k])
        norm_vcX[:, j, k] = rank_utils.normalize(vcX[:, j, k])

#%% Make selection of models based on manual input\
gradient = gradient[:,:,select_models]
perturbation = perturbation[:,:,select_models]
vcX = vcX[:,:,select_models]

rank_gradient = rank_gradient[:,:,select_models]
rank_perturbation = rank_perturbation[:,:,select_models]
rank_vcX = rank_vcX[:,:,select_models]

norm_gradient = norm_gradient[:,:,select_models]
norm_perturbation = norm_perturbation[:,:,select_models]
norm_vcX = norm_vcX[:,:,select_models]

descr_models = descr_models[select_models]
descr_models_short = descr_models_short[select_models]
models = models[select_models]

n_models  = len(select_models)
# %% Plot different dfigures
print('Check manually if inputs are formatted according ot the same standard and that the same models/eelctrodes are used!')

# %% comparing methods and ranking: 7 subplots (one per device). y-axis rank, x-axis electrodes, bars for methods (so 3 bars per electrodes)
sort_index = 2
# Question to be answered by this plot: Which electrode is most important for a specific device, and do the methods agree?
fig_a, ax_a = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_a.suptitle('Rank vs electrode for every model')
ax_a = ax_a.flatten()
for i in range(n_models):
    data = np.stack((rank_gradient[:, 0, i],
                     rank_perturbation[:, 0, i],
                     rank_vcX[:, 0, i]),
                    axis=0)
    legend = descr_methods
    xticklabels = descr_elec
    ax = ax_a[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=2)
    ax.set_title(descr_models_short[i])
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Rank by method')

# %% same as above, but then on the normalized data.
fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_b.suptitle('Normalized score vs electrode for every model')
ax_b = ax_b.flatten()
for i in range(n_models):
    data = np.stack((norm_gradient[:, 0, i],
                     norm_perturbation[:, 0, i],
                     norm_vcX[:, 0, i]),
                    axis=0)
    legend = descr_methods
    xticklabels = descr_elec
    ax = ax_b[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=2)
    ax.set_title(descr_models_short[i])
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Normalized score')

# %% comparable as above with normalized data, bu then plotting in a 2d plane
fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharey=False)
ax_b = ax_b.flatten()
for i in range(n_models):
    x_data = vcX[:, 0, i]
    y_data1, color1 = perturbation[:, 0, i], 'blue'
    y_data2, color2 = gradient[:,0,i], 'red'
    xticklabels = descr_elec
    ax = ax_b[i]
    ax.set_title(descr_models_short[i])
    ax.set_xlabel('vcX capacity')
    # Plot 1
    ax.plot(x_data, y_data1, linestyle='None', marker='+')
    ax.set_ylabel('Perturbation score', color=color1)
    ax.tick_params(axis='y', color=color1)
    # Plot 2
    ax_twin = ax.twinx()
    ax_twin.plot(x_data, y_data2, color=color2, linestyle='None', marker='+')
    ax_twin.set_ylabel('Gradient score', color=color2)
    ax_twin.tick_params(axis='y', color=color2)

# %% one subplot per method to compare if different models agree on the importantness of the device.
# QUestion that can be answered by this plot: Which electrode is most important according ot this method, and do the different models agree?
fig_c, ax_c = plt.subplots(nrows=1, ncols=3, sharey=False)
fig_c.suptitle('Electrode rank vs electrode nummer, bar color per model, subplot per method')
ax_c = ax_c.flatten()
for i, data in enumerate([rank_gradient, rank_perturbation, rank_vcX]):
    data = data[:, 0, :].T  # select one voltage interval
    legend = descr_models_short  # descr_models
    xticklabels = descr_elec
    ax = ax_c[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=0)
    ax.set_title(descr_methods[i])
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Rank')

# %% One fig per eelectrode, showing ranking vs methods
# Question to be answered: For a specific electrode, which method gives highest ranking, and do the models agree?
fig_d, ax_d = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_d.suptitle('Method vs rank, bar color per model, subplot per electrode')
ax_d = ax_d.flatten()
for i in range(n_elec):
    # First dimension of data: rank vs model
    # Second dimeniosn of data: rank vs method
    data = np.stack((rank_gradient[i, 0, :],
                     rank_perturbation[i, 0, :],
                     rank_vcX[i, 0, :]),
                    axis=1)
    legend = descr_models_short  # descr_models
    xticklabels = descr_methods
    ax = ax_d[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(f'electrode {descr_elec[i]}')
    ax.set_xlabel('Method')
    ax.set_ylabel('Rank')

# %% One figure per electrode, basically the inverse of above
# Question to be answered: For a specific electrode, which model get's highest ranking, and do the methodes agree?
fig_e, ax_e = plt.subplots(nrows=2, ncols=4, sharey=True)
fig_e.suptitle('Model vs rank, bar color per method, subplot per electrode')
ax_e = ax_e.flatten()
for i in range(n_elec):
    # First dimension of data: rank vs model
    # Second dimeniosn of data: rank vs method
    data = np.stack((rank_gradient[i, 0, :],
                     rank_perturbation[i, 0, :],
                     rank_vcX[i, 0, :]),
                    axis=1).T
    legend = descr_methods
    xticklabels = descr_models_short
    ax = ax_e[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(f'electrode {descr_elec[i]}')
    ax.set_xlabel('Model #')
    ax.set_ylabel('Rank')
