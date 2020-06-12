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
n_models_old = 7
n_models_new = 3
shape = [n_elec, n_intervals, n_models]
# Looped in order: [input indices, voltage_intervals, torch_model_dict]

# Load npz libraires containing the data. Contains a lot more information for checking data. Not used in this script, feel free to explore.
vc_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_05_29_capacity_loop_7_models_VC2-8_Brains_only\loop_items.npz', allow_pickle=True)

# Manual information about the loops
descr_elec = np.array([0, 1, 2, 3, 4, 5, 6])
descr_intervals = ['full']
descr_models_short_old = np.array(['brains1', 'darwin1', 'darwin2', 'brains2.1', 'brains2.2', 'pinky1','darwin3'])
descr_models_short_new = np.array(['brains1', 'brains2.1', 'brains2.2'])

# %% Load data from npz libraries
# Get VC data from summaries, because it was not saved correctly:
#vc_summaries = vc_lib['summaries']
#n_vcs = 7
#vc = np.full(np.append(shape, n_vcs), np.nan)
#for i in range(n_elec):
#    for j in range(n_models_old):
#        # zeroth dimension hardcoded, because I did not loop over the one voltage intervals in this case
#        vc[i, 0, j, :] = vc_summaries[i, j]['capacity_per_N']


#%% insert into dataframe
descr_vcs = [2, 3, 4, 5, 6, 7, 8]
vc_summaries = vc_lib['summaries']
for i, elec in enumerate(descr_elec):
    for j, model in enumerate(descr_models_short_new):
        vcs_temp = vc_summaries[i, j]['capacity_per_N']
        for k, vc in enumerate(descr_vcs):
            descr = 'vc' + str(vc)
            df_scores.loc[(elec, 'full', model), descr] = vcs_temp[k]
