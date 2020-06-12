# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:31:28 2020

@author: Jochem

Goal:
    - load data from
        - pertrubation ranking
        - gradient ranking
        - VC dimension
    for different models, and for different electrodes
and compare them

"""

# %% Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bspytasks.analysis.electrode_ranking.ranking_utils as rank_utils
def getRowValues(dataframe, index_key):
    assert isinstance(dataframe, pd.DataFrame)
    index = dataframe.index.names.index(index_key)
#    return dataframe.index.levels[index] # this can be used and is faster than ..unique() approach, but it changes the order to alphabetical! Which we do not want.
    return dataframe.index.get_level_values(index).unique()

# %% Defining user variables file locations and selections of what to plot
scores = pd.read_pickle(r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\results_dataframes\scores_new_brains.pkl')
input_elecs = getRowValues(scores, 'input_elec')
input_intervals = getRowValues(scores, 'input_interval')
input_interval = 'full' # select this interval for plotting.
models = getRowValues(scores, 'model')
vcX = 'vc7' # select this vc dimension for plotting
methods = ['grad', 'pert', vcX]

# Manually invert perturbation score, because it seems to be inversely correlated
scores.loc[:, 'pert'] = -scores.loc[:, 'pert']
print('Perturbation score inverted')

# %% Rank all data:
for i, model in enumerate(models):
    for j, method in enumerate(methods):
        df_filter = (slice(None), input_interval, model) # this selects all data generated for one model.
        scores.loc[df_filter, method+'_rank'] = rank_utils.rank_low_to_high(scores.loc[df_filter, method].values)[0]

# Add one to set zero rank to one, start coutning at one. And set dtype.
for j, method in enumerate(methods):
    scores.loc[:, method+'_rank'] = scores.loc[:, method+'_rank'].astype(int)+1

# %% Normalize data
for i, model in enumerate(models):
    for j, method in enumerate(methods):
        df_filter = (slice(None), input_interval, model) # this selects all data generated for one model.
        scores.loc[df_filter, method+'_norm'] = rank_utils.normalize(scores.loc[df_filter, method].values)

# %% comparing methods and ranking: 7 subplots (one per device). y-axis rank, x-axis electrodes, bars for methods (so 3 bars per electrodes)
sort_index = 2
# Question to be answered by this plot: Which electrode is most important for a specific device, and do the methods agree?
fig_a, ax_a = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_a.suptitle('Rank vs electrode for every model')
ax_a = ax_a.flatten()
for i, model in enumerate(models):
    df_filter = (slice(None), input_interval, model) # this selects all data generated for one model.
    data = np.stack((scores.loc[df_filter, methods[0]+'_rank'].values,
                     scores.loc[df_filter, methods[1]+'_rank'].values,
                     scores.loc[df_filter, methods[2]+'_rank'].values),
                    axis=0)
    legend = methods[0:3]
    xticklabels = input_elecs
    ax = ax_a[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=2)
    ax.set_title(models[i])
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Rank by method')

# %% same as above, but then on the normalized data.
fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_b.suptitle('Normalized score vs electrode for every model')
ax_b = ax_b.flatten()
for i, model in enumerate(models):
    df_filter = (slice(None), input_interval, model) # this selects all data generated for one model.
    data = np.stack((scores.loc[df_filter, methods[0]+'_norm'].values,
                     scores.loc[df_filter, methods[1]+'_norm'].values,
                     scores.loc[df_filter, methods[2]+'_norm'].values),
                    axis=0)
    legend = methods[0:3]
    xticklabels = input_elecs
    ax = ax_b[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=2)
    ax.set_title(models[i])
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Normalized score')

# %% comparable as above with normalized data, bu then plotting in a 2d plane
fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_b.suptitle('XY plot of vcX versus gradient/perturbation')
ax_b = ax_b.flatten()
for i, model in enumerate(models):
    df_filter = (slice(None), input_interval, model)
    x_data = scores.loc[df_filter, vcX]
    y_data1, color1 = scores.loc[df_filter, 'pert'], 'blue'       #perturbation[:, 0, i], 'blue'
    y_data2, color2 = scores.loc[df_filter, 'grad'], 'red'      #gradient[:,0,i], 'red'
    xticklabels = input_elecs
    ax = ax_b[i]
    ax.set_title(models[i])
    ax.set_xlabel(vcX+' capacity')
    # Plot 1
    ax.plot(x_data, y_data1, linestyle='None', marker='+')
    ax.set_ylabel('Perturbation score', color=color1)
    ax.tick_params(axis='y', color=color1)
    # Plot 2
    ax_twin = ax.twinx()
    ax_twin.plot(x_data, y_data2, color=color2, linestyle='None', marker='+')
    ax_twin.set_ylabel('Gradient score', color=color2)
    ax_twin.tick_params(axis='y', color=color2)

# FIll the lsat plot with a combination of all data
#model_selection = ('brains2.1','brains2.2')
#model_selection = ('darwin2','darwin3')
model_selection = slice(None)
df_selection = (slice(None), input_interval, model_selection)
x_data = scores.loc[df_selection, vcX]
y_data1, color1 = scores.loc[df_selection, 'pert'], 'blue'
y_data2, color2 = scores.loc[df_selection, 'grad'], 'red'
ax = ax_b[7]
xticklabels = input_elecs
ax.set_title(model_selection)
ax.set_xlabel(vcX+'capacity')
# Plot 1
ax.plot(x_data, y_data1, linestyle='None', marker='+')
ax.set_ylabel('Perturbation score', color=color1)
ax.tick_params(axis='y', color=color1)
# Plot 2
ax_twin = ax.twinx()
ax_twin.plot(x_data, y_data2, color=color2, linestyle='None', marker='+')
ax_twin.set_ylabel('Gradient score', color=color2)
ax_twin.tick_params(axis='y', color=color2)

# %% one subplot per method to compare if different models agree on the importantness of the device.
# QUestion that can be answered by this plot: Which electrode is most important according ot this method, and do the different models agree?
fig_c, ax_c = plt.subplots(nrows=1, ncols=3, sharey=False)
fig_c.suptitle('Electrode rank vs electrode nummer, bar color per model, subplot per method')
ax_c = ax_c.flatten()
for i, data in enumerate([rank_gradient, rank_perturbation, rank_vcX]):
    data = data[:, 0, :].T  # select one voltage interval
    legend = descr_models_short  # descr_models
    xticklabels = descr_elec
    ax = ax_c[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=0)
    ax.set_title(descr_methods[i])
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Rank')

# %% One fig per eelectrode, showing ranking vs methods
# Question to be answered: For a specific electrode, which method gives highest ranking, and do the models agree?
fig_d, ax_d = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_d.suptitle('Method vs rank, bar color per model, subplot per electrode')
ax_d = ax_d.flatten()
for i in range(n_elec):
    # First dimension of data: rank vs model
    # Second dimeniosn of data: rank vs method
    data = np.stack((rank_gradient[i, 0, :],
                     rank_perturbation[i, 0, :],
                     rank_vcX[i, 0, :]),
                    axis=1)
    legend = descr_models_short  # descr_models
    xticklabels = descr_methods
    ax = ax_d[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(f'electrode {descr_elec[i]}')
    ax.set_xlabel('Method')
    ax.set_ylabel('Rank')

# %% One figure per electrode, basically the inverse of above
# Question to be answered: For a specific electrode, which model get's highest ranking, and do the methodes agree?
fig_e, ax_e = plt.subplots(nrows=2, ncols=4, sharey=True)
fig_e.suptitle('Model vs rank, bar color per method, subplot per electrode')
ax_e = ax_e.flatten()
for i in range(n_elec):
    # First dimension of data: rank vs model
    # Second dimeniosn of data: rank vs method
    data = np.stack((rank_gradient[i, 0, :],
                     rank_perturbation[i, 0, :],
                     rank_vcX[i, 0, :]),
                    axis=1).T
    legend = descr_methods
    xticklabels = descr_models_short
    ax = ax_e[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(f'electrode {descr_elec[i]}')
    ax.set_xlabel('Model #')
    ax.set_ylabel('Rank')
