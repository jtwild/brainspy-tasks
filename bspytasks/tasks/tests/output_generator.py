# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:42:15 2020

@author: Jochem
"""

#%% Import packages
from bspyproc.bspyproc import get_processor
from bspyalgo.utils.io import load_configs
import numpy as np
import matplotlib.pyplot as plt

#%% Get processor and get output
configs = load_configs('configs/output_generator.yaml')
processor = get_processor(configs['processor'])
#npz_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\GIT_SourceTree\brainspy-tasks\tmp\input\data\measurement.npz')
npz_lib = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Patch_filter\brains_13032020\validation_run_from_training_2020_03_11_05h19m30s_mark_fill\_2020_03_13_141717\numpydata.npz')
waveform = npz_lib['merged_inputs_waveform']
mask = npz_lib['mask']
output_hardware = npz_lib['output_hardware']

#%% Get output
output_simulation = processor.get_output(waveform)

#%% Take subseletion of output
num_features = 16
output_temp = output_simulation[mask]
xlen = len(output_temp) / num_features
ind = np.arange(xlen/2, num_features*xlen, step=xlen).astype(np.int)
output_verification_simulation = output_temp[ind]

#%% Plot
plt.figure()
plt.plot(output_simulation[mask])
plt.plot(-output_hardware[mask])
plt.legend(['Simulation','Hardware'])

#outpath = r'C:\Users\Jochem\STACK\Daily_Usage\GIT_SourceTree\brainspy-tasks\tmp\output\output_getter\brains_verification_13032020\new_simulation.npz'
#np.save(outpath, mask=mask, output_hardware = output_hardware, outputs_waveform_simulation = output_simulation, merged_inputs_waveform = waveform)