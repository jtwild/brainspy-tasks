# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:41:52 2020

@author: Jochem
"""
# %% Load packages
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# %% User data
base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_19_capacity_loop_voltage_intervals_on_electrode0'
glob_filter = '*/summary_results.pkl'
descrips = [[-1.2, 0.6],[-1.2, -0.6], [-0.6, 0.6],  [-1.2, -0.9], [-0.9, -0.6], [-0.6, -0.3], [-0.3, 0], [0, 0.3], [0, 0.3]]
VC_dims = ['3', '4', '5', '6']
# %% Remaining script
# Load data
glob_query = os.path.join(base_dir, glob_filter)
files = glob(glob_query)

summaries = list()
capacities = list()
correlations = list()
for file in files:
    with open(file, 'rb') as input_file:
        summaries.append(pickle.load(input_file))
    capacities.append(summaries[-1]['capacity_per_N'])
    correlations.append(summaries[-1]['correlation_distrib_per_N'])

capacities = np.array(capacities)
correlations = np.array(correlations)

# Average correlations array
for i in range(correlations.shape[0]):
    for j in range(correlations.shape[1]):
        correlations[i, j] = correlations[i, j].mean()
# %% Plot results
# Capacity vs N
plt.figure()
plt.plot(VC_dims, capacities[3:].T)
plt.legend(descrips[3:])
plt.xlabel('VC Dim.')
plt.ylabel('Capacity (correct configs / total configs)')
plt.title('Capacities vs N for electrode 0 /n different voltage ranges')
plt.grid()

# Correlation
plt.figure()
plt.plot(VC_dims, correlations[3:].T)
plt.legend(descrips[3:-1])
plt.xlabel('VC dim.')
plt.ylabel('Correlation')
plt.title('Correlation vs N for electrode 0 /n different voltage ranges')
plt.grid()