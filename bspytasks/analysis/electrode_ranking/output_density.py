# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:40:23 2020

@author: Jochem

Script to look at the output currents, and see if they are within required ranges.
Also looks at input voltages with same reasoning.
"""

# import packages
from bspyproc.processors.simulation.surrogate import SurrogateModel
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def importEverything(infile):
    inData = np.load(infile, allow_pickle=True)
    for varName in inData:
        globals()[varName] = inData[varName]
def getRowValues(dataframe, index_key):
    assert isinstance(vc_info, pd.DataFrame)
    index = dataframe.index.names.index(index_key)
#    return dataframe.index.levels[index] # this can be used and is faster than ..unique() approach, but it changes the order to alphabetical! Which we do not want.
    return dataframe.index.get_level_values(index).unique()
#%% User data
vc_file = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\results_dataframes\vc_data_old_brains.pkl'
vc_dim = 'vc4'    # index + 2 is the VC dimension because vcs[0] = 2
n_bins = 20 # for histograms
# Which models to use for getting voltage ranges?
base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\used_models_ordered'
suffix_dir = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\model_description.npz')['model_description']

#%% Get full directory for models
models_dir = np.full(suffix_dir.shape, '', dtype=object)
for i, suffix in enumerate(suffix_dir):
    models_dir[i] = os.path.join(base_dir, suffix)
# Get clipping values
info_dict = []
models = []
amplitudes = []
clipping_values = []
volt_ranges = []
for i, model_dir in enumerate(models_dir):
    model = SurrogateModel({'torch_model_dict': model_dir})
    models.append(model)
    info_dict.append(model.info)
    volt_ranges.append( [model.min_voltage, model.max_voltage])
    amplitudes.append(model.amplitude)
    clipping_values.append(model.info['data_info']['clipping_value'])

#%% Import all data and headers from pandas.
vc_info = pd.read_pickle(vc_file)
models = getRowValues(vc_info, 'model')
input_elecs = getRowValues(vc_info, 'input_elec')
input_intervals = getRowValues(vc_info, 'input_interval')
input_interval = 'full' # select this interval for plotting.
vc_dims = getRowValues(vc_info, 'vc_dim')
vc_dim = 8 # select this vc dimension for plotting

#%% Make histograms of output data for VC
#%% density vs output current, line per electrode, subfigure per device.
fig_a, ax_a = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_a.suptitle('Output current histograms for all found solutions')
#fig_a.tight_layout()
ax_a = ax_a.flatten()
for i, model in enumerate(models):
    ax = ax_a[i]
    hist_data = np.array([])
    sample_count = 0
    for j, input_elec in enumerate(input_elecs):
        df_filter = (input_elec, input_interval, model, vc_dim)
        mask = vc_info.loc[df_filter,'found'].flatten().astype(bool) # only look at the found units.
        sample_count += mask.size # for relative nancount
        hist_data = np.append(hist_data, vc_info.loc[df_filter, 'output_current'][mask,:].flatten())
    # Count how many samples are found and out of range
    mask_outside_range = np.logical_or((hist_data < clipping_values[i][0]), (hist_data > clipping_values[i][1])) # add a ask where we look only at the output currents outside of our range.
    # Count how many samples are nan.
    nan_count = vc_info.loc[(slice(None), slice(None), model, vc_dim), 'nancounter'].sum() # sum over all input electrodes, over all intervals. Select this model and this vc_dim
    ax.hist(hist_data, density=True, bins = n_bins)
    ax.set_xlabel('output current (nA)')
    ax.set_ylabel('probability (normalized)')
    ax.set_title(models[i] + f'\n {round(mask_outside_range.sum()/mask_outside_range.size * 100, 1)}% of samples outside range' +
                 f'\n {round(nan_count/sample_count * 100, 1)}% of samples nanned')
    ax.autoscale(enable=False)
    ax.plot([clipping_values[i][0]]*2,[0,1], color='black', linestyle='--')
    ax.plot([clipping_values[i][1]]*2,[0,1], color='black', linestyle='--')
#    ax.legend(['Measurement clipping'])
#    ax.legend(descr_elec)

#%% density vs output current, line per electrode, subfigure per device.
selected_elecs = [0]
fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_b.suptitle(f'Output current histogram when using electrodes {selected_elecs} as input')
ax_b = ax_b.flatten()
for i, model in enumerate(models):
    ax = ax_b[i]
    hist_data = np.array([])
    for j, selected_elec in enumerate(selected_elecs):
        df_filter = (selected_elec, input_interval, model, vc_dim)
        mask = vc_info.loc[df_filter,'found'].flatten().astype(bool) # only look at the found units.

        hist_data = np.append(hist_data,  vc_info.loc[df_filter, 'output_current'][mask,:].flatten()) # flatten this object (an numpy array) to use for histogram
    ax.hist(hist_data, density=True, bins = n_bins)
    ax.set_xlabel('output current (nA)')
    ax.set_ylabel('probability (normalized)')
    ax.set_title(models[i])
    ax.autoscale(enable=False)
    ax.plot([clipping_values[i][0]]*2,[0,1], color='black', linestyle='--')
    ax.plot([clipping_values[i][1]]*2,[0,1], color='black', linestyle='--')
    ax.legend(['Measurement clipping'])
#    ax.legend(descr_elec)

#%% density vs input voltage for electrode 6 , line per electrode, subfigure per device. figure per control electrode
for control_elec in [5,6]:
    fig_c, ax_c = plt.subplots(nrows=2, ncols=4, sharey=False)
    fig_c.suptitle(f'Input voltage density for outputs outside range, for all input electrodes and control electrode {control_elec}.')
    ax_c = ax_c.flatten()
    for i, model in enumerate(models):
        corrected= False
        ax = ax_c[i]
        hist_data = np.array([])
        for j, input_elec in enumerate(input_elecs): #j: which electrode was used as input
            if control_elec != input_elec:
                # we cannot look at the control elec if it was used as an input. So check this condition beforehand

                corrected=False # to check whether we need to correct back
                if control_elec > input_elec:
                    #in this case, the control index is shifted shifted, so do control-1
                    control_elec -= 1 # index shift
                    corrected = True
                df_filter = (input_elec, input_interval, model, vc_dim)
                mask1 = vc_info.loc[df_filter,'found'].flatten().astype(bool) # found filter
                mask2 = np.logical_or((vc_info.loc[df_filter, 'output_current'] < clipping_values[i][0]).any(axis=1), (vc_info.loc[df_filter, 'output_current'] > clipping_values[i][1]).any(axis=1)) # add a ask where we look only at the output currents outside of our range.
                mask = np.logical_and(mask1, mask2) # AND found AND (above maximum OR below minimum)

                hist_data = np.append(hist_data,  vc_info.loc[df_filter, 'control_voltages'][mask,control_elec].flatten()) # flatten this object (an numpy array) to use for histogram
                if corrected:
                    # if we had correcte before, shift back (for overview and for ax labels)
                    control_elec+=1
        ax.hist(hist_data, bins = n_bins)
        ax.set_xlabel(f'input voltage elec {control_elec} (V)')
        ax.set_ylabel('num. of samples')
        ax.set_title(models[i])
        ax.autoscale(enable=False)
        ax.plot([volt_ranges[i][0][control_elec]]*2,[0,1], color='black', linestyle='--')
        ax.plot([volt_ranges[i][1][control_elec]]*2,[0,1], color='black', linestyle='--')
        ax.legend([f'{round(mask.sum()/mask.size * 100, 1)}% of samples out of range'])