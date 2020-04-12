# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:02:19 2020

@author: Jochem
"""
# %% Load packages
import numpy as np
import matplotlib.pyplot as plt
import bspytasks.validation.electrode_ranking.perturbation_utils as pert
from sklearn import linear_model
from bspyalgo.utils.io import load_configs

# %% Gather data
# User variables
configs = load_configs('configs/validation/multi_perturbation_multi_electrodes_configs.json')
electrodes_sets = configs['perturbation']['electrodes_sets']
perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']

# Get rmse, compare to unperturbed simulation output
rmse = pert.get_perturbed_rmse(configs, compare_to_measurement=False, return_error=True)

# %% Visualize results
plt.figure()
for j in range(len(electrodes_sets)):
    plt.plot(perturb_fraction_sets, rmse[:, j], marker='s', linestyle='')  # or use plt.semilogy(..)
plt.xlabel('Perturbation fraction')
plt.ylabel('RMSE (nA)')
plt.title('RMSE scaling: simulated (square markers) and linear fit (solid line)')
#legend_entries = (np.array(['Electrode']*len(electrodes_sets)).flatten().astype(str) + np.array(electrodes_sets).flatten().astype(str) ).tolist()
plt.legend(electrodes_sets)
plt.grid()

# Fitting linear
plt.gca().set_prop_cycle(None)  # reset color cycle, such that we have the same colors for the fitted lines as for the markers
linear_params = np.zeros([2, len(electrodes_sets)])  # to save intercept and slope of linear fit
for j in range(len(electrodes_sets)):
    y = rmse[:, j]
    X = np.c_[perturb_fraction_sets]
    sample_weight = np.ones_like(y)
#        sample_weight[0:11] = 0
    clf = linear_model.LinearRegression(fit_intercept=True).fit(X, y, sample_weight)
    X_test = np.c_[np.arange(min(perturb_fraction_sets), max(perturb_fraction_sets), 0.001)]
    plt.plot(X_test, clf.predict(X_test))
    # Save intercept and linear slope
    linear_params[0, j], linear_params[1, j] = clf.intercept_, clf.coef_[0]

# Ranking electrode importance
pert.rank_low_to_high(electrodes_sets, linear_params[1, :], do_plot=True)
plt.ylabel('Slope (RMSE / noise fraction)')
plt.title('Electrode ranking based on RMSE')
