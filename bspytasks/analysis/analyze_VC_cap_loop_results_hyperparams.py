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
# %% Remaining script
# Load data
importEverything(summary_file)

# %% Plot results
# Mean capacity per attempt
capacity_max_attempts = np.mean(capacities, axis=(1, 2, 3))
capacity_learning_rate = np.mean(capacities, axis=(0, 2, 3))
capacity_loss_function = np.mean(capacities, axis=(0, 1, 3))
capacity_nr_epochs = np.mean(capacities, axis=(0, 1, 2))

legend = np.array(['max_attempts', 'learning_rate', 'loss_function', 'nr_epochs'])
# %% Plot results
plt.figure()
plt.plot(capacity_max_attempts)
plt.plot(capacity_learning_rate)
plt.plot(capacity_loss_function)
plt.plot(capacity_nr_epochs)
plt.legend(legend)
plt.grid()

# %% Load gate gap
glob_query = os.path.join(base_dir, glob_filter)
files = glob(glob_query)
nr_files = len(files)  # should be 54
shape = [3, 3, 2, 3]
gap_mean = np.zeros(shape).flatten()
gap_std = np.zeros(shape).flatten()
for i in trange(nr_files):
    min_gap = np.abs(pd.read_pickle(files[i]).query('found==True')['min_gap'])
    gap_mean[i] = min_gap.mean()
    gap_std[i] = min_gap.std()
gap_mean = gap_mean.reshape(shape)
gap_std = gap_std.reshape(shape)

# %% Mean gate per attempt
gap_mean_max_attempts = np.mean(gap_mean, axis=(1, 2, 3))
gap_std_max_attempts = np.sqrt(np.mean(gap_std**2, axis=(1, 2, 3)))
gap_mean_learning_rate = np.mean(gap_mean, axis=(0, 2, 3))
gap_std_learning_rate = np.sqrt(np.mean(gap_std**2, axis=(0, 2, 3)))
gap_mean_loss_function = np.mean(gap_mean, axis=(0, 1, 3))
gap_std_loss_function = np.sqrt(np.mean(gap_std**2, axis=(0, 1, 3)))
gap_mean_nr_epochs = np.mean(gap_mean, axis=(0, 1, 2))
gap_std_nr_epochs = np.sqrt(np.mean(gap_std**2, axis=(0, 1, 2)))
# %% Plot results
x3 = np.array([0, 1, 2])
x2 = np.array([0, 1])
plt.figure()
plt.bar(x3, gap_mean_max_attempts, yerr=gap_std_max_attempts)
plt.bar(x3 + 3, gap_mean_learning_rate, yerr=gap_std_learning_rate)
plt.bar(x2 + 6, gap_mean_loss_function, yerr=gap_std_loss_function)
plt.bar(x3 + 8, gap_mean_nr_epochs, yerr=gap_std_nr_epochs)
plt.legend(legend)
plt.grid()
plt.xlabel('Index of try')
plt.ylabel('Avg. gap size')

# %% AUtomated


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
            data_std =  np.std(data, axis=axis_selection)
        else:
            data_std = np.sqrt(np.mean(stds**2, axis=axis_selection))
        plt.bar(x, data_mean, yerr=data_std)

        xticks = np.concatenate((xticks, x))
        counter += shape[i] + 1  # leave one blank space
    plt.xticks(xticks, rotation=45)
    plt.legend(legend)
    plt.grid()
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
#%% Select the corrsig
data = capacities[:,:,0,:]
legend = legend = np.array(['max_attempts', 'learning_rate', 'nr_epochs'])
ticklabels  = np.concatenate((max_attempts_list, learning_rate_list, nr_epochs_list))
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams\n loss = corrsig')

#%% Select lr 0.01
data = capacities[:,0,0,:]
legend = legend = np.array(['max_attempts', 'nr_epochs'])
ticklabels  = np.concatenate((max_attempts_list, nr_epochs_list))
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams\n loss = corrsig\nlearning_rate = 0.01')

# Select nr_epochs = 250
data = capacities[:,0,0,0]
legend = legend = np.array(['max_attempts'])
ticklabels  = (max_attempts_list)
bar_plotter(data, legend, ticklabels)
plt.ylabel('Capacity')
plt.title('mean Capacity for different hyperparams\n loss = corrsig\nlearning_rate = 0.01\n nr_epochs = 250')

# selct max_attempts = 5?
# loss = corrsig
# nr_epochs = 250
# lr = 0.01
# then how many perceptron?