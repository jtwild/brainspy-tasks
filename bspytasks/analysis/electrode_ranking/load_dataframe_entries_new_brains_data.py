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
n_models = 3
n_vcs = 7
shape = [n_elec, n_intervals, n_models] #so: the loop went over n_models first, then over n_intervals, then over n_elec

descr_elec = np.array([0, 1, 2, 3, 4, 5, 6])
descr_intervals = ['full']
descr_models_short = np.array(['brains1', 'brains2.1', 'brains2.2'])
descr_methods = np.array(['grad', 'pert', 'vcX'])
descr_vcs = np.array([2,3,4,5,6,7,8])

base_dir = base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\2020_05_29_capacity_loop_7_models_VC2-8_Brains_only'
folder_suffix = r'vc_dimension_8\custom_dataframe.pkl'
vc_selection = 8 # also change line 34

#%% Dataframe creation
#index_vc = pd.MultiIndex.from_product((descr_elec, descr_intervals, descr_models_short, descr_vcs), names=('input_elec', 'input_interval','model','vc_dim'))
#df_vc = df = pd.DataFrame(np.nan, index = index_vc, columns = ['found','accuracy','output_current','control_voltages','input_voltages','correlation_score','loss','input_electrode','input_interval','min_gap','min_output','max_output','loss_function','learning_rate','max_attempts','nr_epochs','targets', 'nancounter'])
#df_vc = df_vc.astype(object) # to store numpy arrays etc.

#%% Get dpaths so pkl panda dataframes
capacity_folder_list = os.listdir(base_dir)
capacity_folder_list = capacity_folder_list[:-2]  # last two files are python and npz file

test_files = [os.path.join(base_dir, capacity_folder, folder_suffix) for capacity_folder in capacity_folder_list]
test_data = [pd.read_pickle(pickle_file) for pickle_file in test_files] # this contains all the dataframes
assert len(test_data) == n_elec*n_intervals*n_models, 'Number of files doesnt match expectation!'


#%% Now subtract the in and output data in the same way order as it was generated
counter = 0
for i in range(n_elec):
    for j in range(n_intervals):
        for k in range(n_models):
            # take 1:-1 to ignore first and last value, because these have NaN entries, they are trivially satisfied.
            #again take a loop, because the data is imported
#            outputs[i,j,k] = np.hstack(test_data[counter]['final_output'][1:-1]).astype(np.float)
#            controls[i,j,k] = np.hstack(test_data[counter]['control_voltages'][1:-1]).astype(np.float)
#            found[i,j,k] = test_data[counter].found[1:-1].values

            tuple_filter = (descr_elec[i], descr_intervals[j], descr_models_short[k], vc_selection)
            # Load all the values
            df_vc.loc[tuple_filter,'accuracy'] =  test_data[counter].accuracy[1:-1].values
            df_vc.loc[tuple_filter, 'output_current'] = np.hstack(test_data[counter]['final_output'][1:-1]).astype(np.float).T
            df_vc.loc[tuple_filter, 'control_voltages'] = np.vstack(test_data[counter]['control_voltages'][1:-1]).astype(np.float)
            df_vc.loc[tuple_filter, 'found'] = test_data[counter].found[1:-1].values
            df_vc.loc[tuple_filter, 'input_voltages'] = np.hstack(test_data[counter]['input_voltages'][1:-1]).astype(np.float).T
            df_vc.loc[tuple_filter, 'correlation_score'] = test_data[counter].correlation[1:-1].values
            df_vc.loc[tuple_filter, 'loss'] = test_data[counter].final_performance[1:-1].values
            df_vc.loc[tuple_filter, 'input_electrode'] = np.hstack(test_data[counter]['input_electrodes'][1:-1]).astype(np.float).T
            df_vc.loc[tuple_filter, 'input_interval'] = np.vstack(test_data[counter]['voltage_intervals'][1:-1]).astype(np.float)
            df_vc.loc[tuple_filter, 'min_gap'] = test_data[counter].min_gap[1:-1].values
            df_vc.loc[tuple_filter, 'max_output'] = test_data[counter].max_output[1:-1].values
            df_vc.loc[tuple_filter, 'min_output'] = test_data[counter].min_output[1:-1].values
            df_vc.loc[tuple_filter, 'loss_function'] = test_data[counter].loss_function[1:-1].values
            df_vc.loc[tuple_filter, 'learning_rate'] = test_data[counter].learning_rate[1:-1].values
            df_vc.loc[tuple_filter, 'max_attempts'] = test_data[counter].max_attempts[1:-1].values
            df_vc.loc[tuple_filter, 'nr_epochs'] = test_data[counter].nr_epochs[1:-1].values
            df_vc.loc[tuple_filter, 'targets'] = np.hstack(test_data[counter]['targets'][1:-1]).astype(np.float).T

            # Get nan counter
            df_vc.loc[tuple_filter, 'nancounter']  = np.any(np.isnan(df_vc.loc[tuple_filter, 'output_current']), axis=1).sum() #check how many values are nan, ON ANY OF THE OUTPUT POINTSs. Experimental observations seems thatusually all points are nan.

            # Increase counter to keep track of test_files flattened list.
            counter += 1