# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:01:14 2020

@author: Jochem

General utilities used for analysis
"""

import matplotlib.pyplot as plt
import numpy as np

# %% Automated bar plotter with std from data or from stds
def bar_plotter(data, legend, xticklabels, stds='auto', ax=None):
    # Goal: plot multiple bar plots next to each other, for all dimensions in the data. Expecting 2 dimensional data.
    assert data.ndim == 2
    if ax == None:
        # Create new axis if not supplied
        plt.figure()
        ax = plt.axes()
    shape = data.shape
    n_dim = data.ndim
    selection_base = range(n_dim)
    counter = 0
    xticks = np.array([])
    for i in range(n_dim):
        axis_selection = tuple(x for x in selection_base if x != i)
        data_mean = np.mean(data, axis=axis_selection)
        x = range(counter, counter + shape[i])
        if np.all(stds == 'auto'):
            data_std = np.std(data, axis=axis_selection)
        else:
            data_std = np.sqrt(np.mean(stds**2, axis=axis_selection))
        plt.bar(x, data_mean, yerr=data_std)

        xticks = np.concatenate((xticks, x))
        counter += shape[i] + 1  # leave one blank space
    plt.xticks(xticks, rotation=45)
    plt.legend(legend, loc='lower right')
    plt.grid(b=True, axis='y')
    plt.gca().set_xticklabels(xticklabels)

def sort_by_input_voltage(inputs, values, min_val=None, max_val=None, granularity=None):
    # Sort the values by ranges in inputs. Inputs gets grouped, and indices specifying to
    # a specific group are taken together from the values array
    # inputs should be shape (X,), values should be (X,),
    # Output-wise: values_subsets will be an array of arrays, grid gives the centrepoints of ordering
    # and ranges gives the edges which are used for ordering
    if [min_val, max_val, granularity] == [None, None, None]:
        # If no values supplied:
        min_val = inputs.min()
        max_val = inputs.max()
        granularity = (max_val - min_val) / 5  # by default, group in 5 groups
    grid = np.arange(min_val, max_val, granularity)
    ranges = np.concatenate(([min_val], (grid[1:] + grid[:-1]) / 2, [max_val]))
    values_subsets = [[]] * len(grid)  # no preallocation because size is unknown
    for i in range(len(grid)):
        extraction_condition = np.logical_and(inputs >= ranges[i], inputs < ranges[i + 1])
        values_subsets[i] = np.extract(extraction_condition, values)
    return np.array(values_subsets), grid, ranges


def plot_hist(values, ax=None, n_bins=15, xlabel=''):
    # Plot a single histogram in a specific axes
    if ax == None:
        fig, ax = plt.figure()
    ax.hist(values, histtype='step', density=True, bins=n_bins)
    ax.set_ylabel('Relative probability')
    ax.set_xlabel(xlabel)


def plot_hists(values, ax=None, n_bins=15, legend=None):
    # Plot multiple histograms, potentially in a subplot
    if ax == None:
        plt.figure()
        ax = plt.axes()
    for i in range(len(values)):
        plot_hist(values[i], ax=ax, n_bins=n_bins)
    if all(legend) != None:
        ax.legend(legend)


def rank_low_to_high(values, descriptions = [], plot_type=None, ax=None, x_data = []):
    # Ranks the descriptions according to the values, and potentially makes a barplot out of it.
    # Potentially plots in a specified axes

    values = np.array(values)
    ranking_indices = np.argsort(-values)  # take negative of value to order the values from largest (most positive -> most negative) to smallest
    ranked_values = values[ranking_indices]
    # Then check if we have gotten any descriptions:
    if len(descriptions != 0):
        descriptions = np.array(descriptions)
        ranked_descriptions = descriptions[ranking_indices]
    # Plot ranking in barplot
    if plot_type!=None:
        if ax == None:
            plt.figure()
            ax = plt.axes()
        if len(x_data) == 0:
            x_data = np.arange(len(values))
            #else, use supplied x_data
        width = (x_data[1:] - x_data[:-1]) * 0.8
        width = np.append(width,width[-1]) # make last interval as big as previous one
        if plot_type == 'values':
            ax.bar(x_data, ranked_values, width=width)
            for i, x_val in enumerate(x_data):
                s = np.array2string(np.array(ranking[i]))
                xy = [x_val, ranked_values[i]]
                ax.annotate(s, xy)
            ax.set_xlabel('Ranking')
        elif plot_type == 'ranking':
            y_ticks = np.arange(0, len(x_data))
            y_data = y_ticks[-1::-1]
            ax.bar(x_data[ranking_indices], y_data, width=width) #negative range, ebcause first element is most important
            ax.set_ylabel('Ranking')
            ax.set_yticks(y_data)
            ax.set_yticklabels(y_ticks)
        ax.grid(True)
    elif plot_type==None:
        print('Plot intentionally skipped')
    else:
        NotImplementedError("Plottype not implemented. Choose from 'old', 'new' or None")
    # Ranked descriptions: the input descriptions ranked in the high-to-low order determined by values (might become optional in the ruture)
    # Ranked values: the input values, ordered high to low
    # Ranking indices: the indices that are used to rank the values high to low. So, of the 6th element of value-array has the largest value, the first element
    # of ranking_indices will be 6. If the 0th element of values-array has smallest number, the last element of ranking_indices will be 0. So somethign like [6, ..., 0]
    if len(descriptions) != 0:
        return ranked_values, ranking_indices, ranked_descriptions
    else:
        return ranked_values, ranking_indices


def np_object_array_mean(obj_arr, nan_val = 0):
    # Takes the averages of the object elements of a numpy array.
    # Because if you have a numpy array of a numpy array, just using np.mean(arr, axis=0) does not work.
    float_arr = np.zeros(obj_arr.size)
    for i in range(obj_arr.size):
        float_arr[i] = obj_arr[i].mean()
        if np.isnan(float_arr[i]):
            float_arr[i] = nan_val
    return float_arr
