# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:46:57 2020

@author: Jochem
"""
import os
import pandas as pd
import numpy as np

#%% Information for importing
n_elec = 7  # Hardcoded because all data must match this
n_intervals = 1
n_models = 7
n_vcs = 7
shape = [n_elec, n_intervals, n_models] #so: the loop went over n_models first, then over n_intervals, then over n_elec

descr_elec = np.array([0, 1, 2, 3, 4, 5, 6])
descr_models_short = np.array(['brains1', 'darwin1', 'darwin2', 'brains2.1', 'brains2.2', 'pinky1','darwin3'])
descr_methods = np.array(['grad', 'pert', 'vcX'])
descr_vcs = np.array([2,3,4,5,6,7,8])

base_dir = base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_04_29_capacity_loop_7_models_VC2-6'
folder_suffix = r'vc_dimension_6/custom_dataframe.pkl'

#%% Get dpaths so pkl panda dataframes
capacity_folder_list = os.listdir(base_dir)
capacity_folder_list = capacity_folder_list[:-3]  # last two files are python and npz file

test_files = [os.path.join(base_dir, capacity_folder, folder_suffix) for capacity_folder in capacity_folder_list]
test_data = [pd.read_pickle(pickle_file) for pickle_file in test_files] # this contains all the dataframes
assert len(test_data) == n_elec*n_intervals*n_models, 'Number of files doesnt match expectation!'

#%% Now subtract the in and output data in the same way order as it was generated
counter = 0
outputs = np.full(shape, np.nan, dtype=object) #need to use object arrays because not all VCs= results have the same size
controls = np.full(shape, np.nan, dtype=object)
found = np.full(shape, np.nan, dtype=object)
for i in range(n_elec):
    for j in range(n_intervals):
        for k in range(n_models):
            # take 1:-1 to ignore first and last value, because these have NaN entries, they are trivially satisfied.
            #again take a loop, because the data is imported
            outputs[i,j,k] = np.hstack(test_data[counter]['final_output'][1:-1]).astype(np.float)
            controls[i,j,k] = np.hstack(test_data[counter]['control_voltages'][1:-1]).astype(np.float)
            found[i,j,k] = test_data[counter].found[1:-1].values
            nancounter[i,j,k] = np.any(np.isnan(outputs[i,j,k]), axis=0).sum()) #check how many values are nan, ON ANY OF THE OUTPUT POINTSs. Experimental observations seems thatusually all points are nan.
            counter += 1