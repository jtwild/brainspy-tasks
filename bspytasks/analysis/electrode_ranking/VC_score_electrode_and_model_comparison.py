# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:18:33 2020

@author: Jochem
"""

import numpy as np
import matplotlib.pyplot as plt

#%% LOad data
vc_lib = np.load('C:\\Users\\Jochem\\STACK\\Daily_Usage\\Bestanden\\UT\\TN_MSc\\Afstuderen\\Results\\Electrode_importance\\2020_04_29_Models_Electrodes_Comparison\\vc2-8.npz')

vc = vc_lib['vc']
descr_shape = vc_lib['descr_shape']
descr_elec = vc_lib['descr_elec']
n_elec = len(descr_elec)
descr_intervals = vc_lib['descr_intervals']
n_intervals = len(descr_intervals)
descr_models = vc_lib['descr_models']
n_models = len(descr_models)
descr_vcs = vc_lib['descr_vcs']
n_vcs = len(descr_vcs)
descr_models_short = vc_lib['descr_models_short']

#%% Plotting data
#%% VC vs capacity, line per electrode, subfigure per device.
fig_a, ax_a = plt.subplots(nrows=2, ncols=4, sharey=False)
ax_a = ax_a.flatten()
for i in range(n_models):
    y_data = vc[:,0,i,:].T
    x_data = descr_vcs
    xticklabels = descr_vcs
    ax = ax_a[i]
    ax.set_title(descr_models_short[i])
    ax.set_xlabel('vc dimension')
    ax.set_ylabel('capacity')
    # Plot 1
    ax.plot(x_data, y_data, marker='x')
    ax.legend(descr_elec)
#%% VC vs capacity, line per model, subfigure per electrode.
fig_a, ax_a = plt.subplots(nrows=2, ncols=4, sharey=False)
ax_a = ax_a.flatten()
for i in range(n_elec):
    y_data = vc[i,0,:,:].T
    x_data = descr_vcs
    xticklabels = descr_vcs
    ax = ax_a[i]
    ax.set_title('electrode ' + str(descr_elec[i]))
    ax.set_xlabel('vc dimension')
    ax.set_ylabel('capacity')
    # Plot 1
    ax.plot(x_data, y_data, marker='x')
    ax.legend(descr_models_short)
