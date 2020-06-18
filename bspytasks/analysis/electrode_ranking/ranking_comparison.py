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

# %% User variables
scores_file = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\results_dataframes\scores_new_brains.pkl'
spread_file = r'C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\Results\Electrode_importance\2020_04_29_Models_Electrodes_Comparison\results_dataframes\vc_spread_data.pkl'
n_plot_rows = 2
n_plot_cols = 3

# %%Load plot variables. POssible user selection
scores = pd.read_pickle(scores_file)
scores = scores.drop(labels='darwin1',level='model') # drop darwin1 because the model is bad
input_elecs = getRowValues(scores, 'input_elec')
input_intervals = getRowValues(scores, 'input_interval')
input_interval = 'full' # select this interval for plotting.
models = getRowValues(scores, 'model') # all models
vcX = 'vc7' # select this vc dimension for plotting
methods = ['grad', 'pert', vcX]
#methods = ['vc2','vc3','vc4','vc5','vc6','vc7','vc8']

# Manually invert perturbation score, because it seems to be inversely correlated
scores.loc[:, 'pert'] = -scores.loc[:, 'pert']
print('Perturbation score inverted')

# get n of ....
n_methods = len(methods)
n_input_intervals = len(input_intervals)
n_input_elecs = len(input_elecs)
n_models = len(models)

# Get errorbar information
spread_data = pd.read_pickle(spread_file)
spread = pd.DataFrame(spread_data.loc[(slice(None), input_interval, 'darwin2', slice(None), (0, 1, 2, 3)), 'capacity'].astype(float).mean(axis=0, level=['input_elec','input_interval','model','vc_dim']))
spread.rename({'capacity': 'avg'}, axis=1, inplace=True)
spread.loc[:, 'max'] = spread_data.loc[(slice(None), input_interval, 'darwin2', slice(None), (0, 1, 2, 3)), 'capacity'].astype(float).max(axis=0, level=['input_elec','input_interval','model','vc_dim'])
spread.loc[:, 'min'] = spread_data.loc[(slice(None), input_interval, 'darwin2', slice(None), (0, 1, 2, 3)), 'capacity'].astype(float).min(axis=0, level=['input_elec','input_interval','model','vc_dim'])
spread.loc[:, 'error'] = spread.loc[:, 'max'] - spread.loc[:, 'min']
spread.loc[:, 'error_high'] = spread.loc[:, 'max'] - spread.loc[:, 'avg']
spread.loc[:, 'error_low'] = spread.loc[:, 'avg'] - spread.loc[:, 'min']
# %% Rank all data:
for model in models:
    for method in methods:
        df_filter = (slice(None), input_interval, model) # this selects all data generated for one model.
        scores.loc[df_filter, method+'_rank'] = rank_utils.rank_low_to_high(scores.loc[df_filter, method].values)[0]

# Add one to set zero rank to one, start coutning at one. And set dtype.
for method in methods:
    scores.loc[:, method+'_rank'] = scores.loc[:, method+'_rank'].astype(int)+1

# %% Normalize data
for model in models:
    for method in methods:
        df_filter = (slice(None), input_interval, model) # this selects all data generated for one model.
        scores.loc[df_filter, method+'_norm'] = rank_utils.normalize(scores.loc[df_filter, method].values)

# %% comparing methods and ranking: 7 subplots (one per device). y-axis rank, x-axis electrodes, bars for methods (so 3 bars per electrodes)
sort_index = 2
# Question to be answered by this plot: Which electrode is most important for a specific device, and do the methods agree?
fig_a, ax_a = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, sharey=False)
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
    ax.set_title(model)
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Rank by method')

# %% same as above, but then on the normalized data.
fig_b, ax_b = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, sharey=False)
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
    ax.set_title(model)
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Normalized score')

# %% comparable as above with normalized data, bu then plotting in a 2d plane
fig_b, ax_b = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, sharey=False)
fig_b.suptitle('XY plot of vcX versus gradient/perturbation')
ax_b = ax_b.flatten()
for i, model in enumerate(models):
    df_filter = (slice(None), input_interval, model)
    x_data = scores.loc[df_filter, vcX]
    y_data1, color1 = scores.loc[df_filter, 'pert'], 'blue'
    y_data2, color2 = scores.loc[df_filter, 'grad'], 'red'
    xticklabels = input_elecs
    ax = ax_b[i]
    ax.set_title(model)
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

## FIll the lsat plot with a combination of all data
##model_selection = ('brains2.1','brains2.2')
##model_selection = ('darwin2','darwin3')
#model_selection = slice(None)
#df_selection = (slice(None), input_interval, model_selection)
#x_data = scores.loc[df_selection, vcX]
#y_data1, color1 = scores.loc[df_selection, 'pert'], 'blue'
#y_data2, color2 = scores.loc[df_selection, 'grad'], 'red'
#ax = ax_b[7]
#xticklabels = input_elecs
#ax.set_title(model_selection)
#ax.set_xlabel(vcX+'capacity')
## Plot 1
#ax.plot(x_data, y_data1, linestyle='None', marker='+')
#ax.set_ylabel('Perturbation score', color=color1)
#ax.tick_params(axis='y', color=color1)
## Plot 2
#ax_twin = ax.twinx()
#ax_twin.plot(x_data, y_data2, color=color2, linestyle='None', marker='+')
#ax_twin.set_ylabel('Gradient score', color=color2)
#ax_twin.tick_params(axis='y', color=color2)

# %% one subplot per method to compare if different models agree on the importantness of the device.
# QUestion that can be answered by this plot: Which electrode is most important according ot this method, and do the different models agree?
fig_c, ax_c = plt.subplots(nrows=1, ncols=n_methods, sharey=False)
fig_c.suptitle('Electrode rank vs electrode nummer, bar color per model, subplot per method')
ax_c = ax_c.flatten()
for i, method in enumerate(methods):
    data = np.zeros((n_models, n_input_elecs))
    for j, model in enumerate(models):
        data[j,:] = scores.loc[(slice(None), input_interval, model), method+'_rank'].values
    legend = models
    xticklabels = input_elecs
    ax = ax_c[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax, sort_index=0)
    ax.set_title(method)
    ax.set_xlabel('Electrode #')
    ax.set_ylabel('Rank')

# %% One fig per eelectrode, showing ranking vs methods
# Question to be answered: For a specific electrode, which method gives highest ranking, and do the models agree?
fig_d, ax_d = plt.subplots(nrows=2, ncols=4, sharey=False)
fig_d.suptitle('Method vs rank, bar color per model, subplot per electrode')
ax_d = ax_d.flatten()
for i, input_elec in enumerate(input_elecs):
    # First dimension of data: rank vs model
    # Second dimeniosn of data: rank vs method
    df_filter = (input_elec, input_interval, slice(None)) # this selects all data generated for one model.
    data = np.stack((scores.loc[df_filter, methods[0]+'_rank'].values,
                     scores.loc[df_filter, methods[1]+'_rank'].values,
                     scores.loc[df_filter, methods[2]+'_rank'].values),
                    axis=1)
    legend = models  # descr_models
    xticklabels = methods
    ax = ax_d[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(f'electrode {input_elec}')
    ax.set_xlabel('Method')
    ax.set_ylabel('Rank')

# %% One figure per electrode, basically the inverse of above
# Question to be answered: For a specific electrode, which model get's highest ranking, and do the methodes agree?
fig_e, ax_e = plt.subplots(nrows=2, ncols=4, sharey=True)
fig_e.suptitle('Model vs rank, bar color per method, subplot per electrode')
ax_e = ax_e.flatten()
for i, input_elec in enumerate(input_elecs):
    # First dimension of data: rank vs model
    # Second dimeniosn of data: rank vs method
    df_filter = (input_elec, input_interval, slice(None)) # this selects all data generated for one model.
    data = np.stack((scores.loc[df_filter, methods[0]+'_rank'].values,
                     scores.loc[df_filter, methods[1]+'_rank'].values,
                     scores.loc[df_filter, methods[2]+'_rank'].values),
                    axis=0)
    legend = methods
    xticklabels = models
    ax = ax_e[i]
    rank_utils.bar_plotter_2d(data, legend, xticklabels, ax=ax)
    ax.set_title(f'electrode {input_elec}')
    ax.set_xlabel('Model #')
    ax.set_ylabel('Rank')

# %% Plot Capacity with spread
# subfigure per device, line per electrode, X-axis = vc dim, y-axis = capacity
# First only for darwin
model = 'darwin2'
deltas = np.linspace(-0.15, 0.15, 7)
vcs = getRowValues(spread, 'vc_dim')

# Create figure
fig_f, ax_f = plt.subplots(nrows=1, ncols = 2, tight_layout=True)
#fig_f.suptitle('Capacity vs dimension')
ax_f = ax_f.flatten()
ax = ax_f[0]
ax.set_xlabel('VC dimension')
ax.set_xticks(vcs)
ax.set_ylabel('Capacity')
ax.grid(b='True')
for i, input_elec in enumerate(input_elecs):
    score_filter = (input_elec, input_interval, model)
    spread_filter = (input_elec, input_interval, model, slice(None))
    y_data = spread.loc[spread_filter, 'avg']
    y_error = [spread.loc[spread_filter, 'error_low'], spread.loc[spread_filter, 'error_high']]
    x_data = vcs + deltas[i]

    caplines = ax.errorbar(x_data, y_data, yerr=y_error, lolims=True, uplims=True)[1]
    caplines[0].set_marker('_')
    caplines[1].set_marker('_')
    ax.set_title('(a)')
    ax.legend(['elec. '+ str(x) for x in input_elecs])
#    caplines[0].set_markersize(10)

ax = ax_f[1]
ax.set_title('(b)')
hist_data = spread.loc[:,'error']
ax.hist(hist_data, cumulative=True, density=True, bins=100)
ax.set_xlim([hist_data.min(), hist_data.max()])
ax.set_ylim([0, 1])
ax.grid(b='True')
ax.set_xlabel('Spread in capacity')
ax.set_ylabel('Fraction of points with lower spread')

plt.tight_layout()
