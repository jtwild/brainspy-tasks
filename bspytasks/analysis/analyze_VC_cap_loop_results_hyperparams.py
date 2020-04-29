# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:41:52 2020

@author: Jochem
"""
# %% Load packages
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import trange
import pickle


def importEverything(infile):
    # Only use trusted files with allow_pickle=True
    inData = np.load(infile, allow_pickle=True)
    for varName in inData:
        globals()[varName] = inData[varName]


# %% User data
# Capacity summary
summary_file = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_21_capacity_loop_hyperparameters\loop_items.npz'
# Other information, including gap size
base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_21_capacity_loop_hyperparameters\capacity_data'
glob_filter = '*/vc_dimension_5/custom_dataframe.pkl'
# depending on which files are loaded:
legend = np.array(['max_attempts', 'learning_rate', 'loss_function', 'nr_epochs'])
shape = [3, 3, 2, 3]
# %% Remaining script
# Load data
importEverything(summary_file)


# %% Load gate gap
glob_query = os.path.join(base_dir, glob_filter)
files = glob(glob_query)
nr_files = len(files)
gap_mean = np.zeros(shape).flatten()
gap_std = np.zeros(shape).flatten()
for i in trange(nr_files):
    min_gap = np.abs(pd.read_pickle(files[i]).query('found==True')['min_gap'])
    gap_mean[i] = min_gap.mean()
    gap_std[i] = min_gap.std()
gap_mean = gap_mean.reshape(shape)
gap_std = gap_std.reshape(shape)


# %% Automated bar plotter with std from data or from stds
def bar_plotter(data, legend, xticklabels, stds='auto'):
    plt.figure()
    shape = data.shape
    n_dim = data.ndim
    selection_base = range(n_dim)
    counter = 0
    xticks = np.array([])
    for i in range(n_dim):
        axis_selection = tuple(x for x in selection_base if x != i)
        data_mean = np.mean(data, axis=axis_selection)
        x = range(counter, counter + shape[i])
        if np.all(stds == 'auto'):
            data_std = np.std(data, axis=axis_selection)
        else:
            data_std = np.sqrt(np.mean(stds**2, axis=axis_selection))
        plt.bar(x, data_mean, yerr=data_std)

        xticks = np.concatenate((xticks, x))
        counter += shape[i] + 1  # leave one blank space
    plt.xticks(xticks, rotation=45)
    plt.legend(legend, loc='lower right')
    plt.grid(b=True, axis='y')
    plt.gca().set_xticklabels(xticklabels)


# %% AUtomate barplot tester
# Gap mean
data = gap_mean
stds = gap_std
ticklabels = np.concatenate((max_attempts_list, learning_rate_list, loss_function_list, nr_epochs_list))
bar_plotter(data, legend, ticklabels, stds)
plt.ylabel('Avg. gap size with std. (nA)')
plt.title('Avg. gap size for different hyperparams')
# Capacity
data = capacities
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams')
# %% Select the corrsig
data = capacities[:, :, 0, :]
legend = legend = np.array(['max_attempts', 'learning_rate', 'nr_epochs'])
ticklabels = np.concatenate((max_attempts_list, learning_rate_list, nr_epochs_list))
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams\n loss = corrsig')

# %% Select lr 0.01
data = capacities[:, 0, 0, :]
legend = legend = np.array(['max_attempts', 'nr_epochs'])
ticklabels = np.concatenate((max_attempts_list, nr_epochs_list))
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams\n loss = corrsig\nlearning_rate = 0.01')

# Select nr_epochs = 250
data = capacities[:, 0, 0, 0]
legend = legend = np.array(['max_attempts'])
ticklabels = (max_attempts_list)
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams\n loss = corrsig\nlearning_rate = 0.01\n nr_epochs = 250')

# selct max_attempts = 5?
# loss = corrsig
# nr_epochs = 250
# lr = 0.01
# then how many perceptron?


#%% Look at gap quality for specific scenario
data = gap_mean[2,0,0,:]
stds = gap_std[2,0,0,:]
legend = legend = np.array(['nr_epochs'])
ticklabels = (nr_epochs_list)
bar_plotter(data, legend, ticklabels, stds)