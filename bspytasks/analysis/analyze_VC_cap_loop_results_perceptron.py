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
from tqdm import trange
import pickle
def importEverything(infile):
    inData = np.load(infile, allow_pickle=True)
    for varName in inData:
        globals()[varName] = inData[varName]
# %% User data
base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_21_capacity_loop_hyperparameters\perceptron_data\1000epochs'
loop_items_file = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_21_capacity_loop_hyperparameters\loop_items.npz'
glob_filter = '*.npz'
# %% Remaining script
# Load descriptions
importEverything(loop_items_file)

# Load perceptron data
glob_query = os.path.join(base_dir, glob_filter)
files = glob(glob_query)
nr_epochs = 1000
nr_files = len(files)
perceptron_data = np.zeros([nr_files, nr_epochs])
for i in trange(nr_files):
    perceptron_data[i, :] = np.load(files[i])['accuracy_array']

#%% FInd convergence
equal_array = perceptron_data[:,:-1] == perceptron_data[:,1:]
converged = np.zeros(nr_files)
mask = np.where(perceptron_data[:,-1]==100)[0] # select only solutions where 100% accuracy was obtained: akak a solution was found
equal_array[:,0] = False # if this is the only false in the list, then perceptron could not make it better.
for i in trange(nr_files):
    converged[i] = np.where(equal_array[i,:] == False)[0][-1] # the first return element: [0] is the index, and then we want the last one: [-1]
plt.figure()
plt.hist(converged[mask], bins=1000, cumulative=True, density=True)
plt.xlim([0, 1000])
plt.ylim([0, 1])
plt.grid()
plt.title('Fraction of found solutions to VC5 which are converged')
plt.xlabel('Perceptron epoch')
plt.ylabel('Fraction of solutions which are converged')
