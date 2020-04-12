# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:02:19 2020

@author: Jochem
"""
# %% Load packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import bspytasks.validation.electrode_ranking.perturbation_utils as pert
from sklearn import linear_model
from bspyalgo.utils.io import load_configs

# %% Gather data
# User variables
configs = load_configs('configs/validation/multi_perturbation_multi_electrodes_configs.json')
electrodes_sets = configs['perturbation']['electrodes_sets']
perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']

# Get rmse, compare to unperturbed simulation output (not to measurement output)
error = pert.get_perturbed_rmse(configs, compare_to_measurement=False, return_error=True)[1]
error = error.squeeze(axis=1)  # remove the electrodes_sets axis
# %% Create animated plot
# User variables
fig, ax = plt.subplots()
frames = np.logspace(0, 4, 100, dtype=int)

# Do the animation
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
assert len(electrodes_sets) == 1
begin = 10000
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')


def init():
    ax.set_xlabel('Perturbation fraction')
    ax.set_ylabel('Normalized RMSE')
    ax.set_xlim([0, max(perturb_fraction_sets)])
    ax.set_ylim([0, 1.2])
    ax.plot(ax.get_xlim(), [0, 1], color='black')  # line to converge to
    ax.grid(b=True)
    plt.show()
    return ln, text


def update(frame):
    xdata = (perturb_fraction_sets)
    ydata = (np.sqrt(np.mean(error[:, begin:begin + frame]**2, axis=1)))
    ln.set_data(xdata, ydata / ydata[-1])
    text.set_text(f'Number of averaged points: {frame}')
    return ln, text


ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True)
# Saving the animation has dependency:
# conda install -c conda-forge ffmpeg
ani.save(r'tmp/output/perturbation_animation.mp4')
plt.show()