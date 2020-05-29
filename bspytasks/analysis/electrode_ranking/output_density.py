# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:40:23 2020

@author: Jochem
"""

# import packages
from bspyproc.processors.simulation.surrogate import SurrogateModel
import os
import numpy as np
import matplotlib.pyplot as plt
def importEverything(infile):
    inData = np.load(infile, allow_pickle=True)
    for varName in inData:
        globals()[varName] = inData[varName]

#%% User data
IOfile = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\input_output_data.npz'
vc_index = 4    # index - 2 is the VC dimension because vcs[0] = 2
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

#%% Import voltage and curernt data
    # note: bit ugly because our editor cannot see which vairbales get imported, so it will throw some warnings
importEverything(IOfile)

#%% Make histograms of output data for VC
#%% density vs output current, line per electrode, subfigure per device.
fig_a, ax_a = plt.subplots(nrows=2, ncols=4, sharey=False)
ax_a = ax_a.flatten()
for i in range(n_models):
    ax = ax_a[i]
    hist_data = np.array([])
    for j in range(n_elec):
        mask = found[j,0,i,vc_index].flatten().astype(bool) # only look at the found solutions
        hist_data = np.append(hist_data,  outputs[j,0,i,vc_index][:,mask].flatten()) # flatten this object (an numpy array) to use for histogram
    ax.hist(hist_data, density=True, bins = n_bins)
    ax.set_xlabel('output current (nA)')
    ax.set_ylabel('probability (normalized)')
    ax.set_title(descr_models_short[i])
    ax.autoscale(enable=False)
    ax.plot([clipping_values[i][0]]*2,[0,1], color='black', linestyle='--')
    ax.plot([clipping_values[i][1]]*2,[0,1], color='black', linestyle='--')
    ax.legend(['Measurement clipping'])
#    ax.legend(descr_elec)

#%% density vs output current, line per electrode, subfigure per device.
selected_elec = [6]
fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharey=False)
ax_b = ax_b.flatten()
for i in range(n_models):
    ax = ax_b[i]
    hist_data = np.array([])
    for j in selected_elec:
        mask = found[j,0,i,vc_index].flatten().astype(bool) # only look at the found solutions
        hist_data = np.append(hist_data,  outputs[j,0,i,vc_index][:,mask].flatten()) # flatten this object (an numpy array) to use for histogram
    ax.hist(hist_data, density=True, bins = n_bins)
    ax.set_xlabel('output current (nA)')
    ax.set_ylabel('probability (normalized)')
    ax.set_title(descr_models_short[i] + f' elec {selected_elec} ')
    ax.autoscale(enable=False)
    ax.plot([clipping_values[i][0]]*2,[0,1], color='black', linestyle='--')
    ax.plot([clipping_values[i][1]]*2,[0,1], color='black', linestyle='--')
    ax.legend(['Measurement clipping'])
#    ax.legend(descr_elec)

#%% density vs input voltage for electrode 6 , line per electrode, subfigure per device.
for control_elec in [0,1,2,3,4,5,6]:
    fig_c, ax_c = plt.subplots(nrows=2, ncols=4, sharey=False)
    ax_c = ax_c.flatten()
    for i in range(n_models):
        corrected= False
        ax = ax_c[i]
        hist_data = np.array([])
        for input_elec in range(n_elec): #j: which electrode was used as input
            if control_elec != input_elec:
                # we cannot look at the control elec if it was used as an input. So check this condition beforehand

                corrected=False # to check whether we need to correct back
                if control_elec > input_elec:
                    #in this case, the control index is shifted shifted, so do control-1
                    control_elec -= 1
                    corrected = True
                mask = found[input_elec,0,i,vc_index].flatten().astype(bool) # only look at the found solutions
                hist_data = np.append(hist_data,  controls[input_elec,0,i,vc_index][mask, control_elec].flatten()) # flatten this object (an numpy array) to use for histogram
                if corrected:
                    # if we had correcte before, shift back (for overview and for ax labels)
                    control_elec+=1
        ax.hist(hist_data, bins = n_bins, density=True)
        ax.set_xlabel(f'input voltage elec {control_elec} (V)')
        ax.set_ylabel('probability (normalized)')
        ax.set_title(descr_models_short[i])
        ax.autoscale(enable=False)
        ax.plot([volt_ranges[i][0][control_elec]]*2,[0,1], color='black', linestyle='--')
        ax.plot([volt_ranges[i][1][control_elec]]*2,[0,1], color='black', linestyle='--')
    #    ax.legend(descr_elec)






