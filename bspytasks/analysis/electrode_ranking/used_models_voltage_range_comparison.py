# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:16:34 2020

@author: Jochem
"""

from bspyproc.processors.simulation.surrogate import SurrogateModel
import numpy as np
import torch
import glob
import os

#%% Which models to use?
base_dir = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\used_models_ordered'
suffix_dir = np.load(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\model_description.npz')['model_description']

#%% Get full directory
models_dir = np.full(suffix_dir.shape, '', dtype=object)
for i, suffix in enumerate(suffix_dir):
    models_dir[i] = os.path.join(base_dir, suffix)

#%% Get models information
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

