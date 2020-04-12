"""
First version on perturbed inputs:
 Noise ('perturbation') is added to the given electrodes, and then it is visualized how the RMSE changes. User variables are
- electrodes, which will be perturbed
- cofnig file which contains the model and corresponding measurement test data
- perturb_fraction which determines how much noise is added relative to the magntiude of the input voltage range. If the input voltages are within -1.2,+0.6V, then an perturb_fraction of 0.1 means 10% of this range is used, so 0.18V. This 0.18V is used to generate noise evenly distributed in the interval [-0.18/2, +0.18/2]. So in the interval [-0.09, +0.09]V.

Author: Jochem Wildeboer
"""
import torch
import numpy as np
import matplotlib.pyplot as plt  # for barplot
from bspyproc.processors.simulation.surrogate import SurrogateModel
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.io import load_configs
from bspysmg.model.data.outputs import test_model

# %% Below are all the functions used for this purpose

def load_data(configs, steps=1):
    # Load input data
    input_file = configs["data"]['input_data_file']
#    inputs, outputs, info = test_model.load_data(input_file, steps)
    return test_model.load_data(input_file, steps)

def perturb_data(configs, inputs_unperturbed, save_data=False, steps=1):
    # Adds noise to a specific electrode of a set of input data
    # Noise gets added to electrodes in configs['perturbation']['electrodes']
    # with an peak-peak amplitude of configs['perturbation']['perturb_fraction'] *100% of
    # that electrodes current value range

    # Load config data
    electrodes = configs['perturbation']['electrodes']
    perturb_fraction = configs['perturbation']['perturb_fraction']

    # Perturb the data of the required electrodes
    inputs_perturbed = np.copy(inputs_unperturbed)
    for i in electrodes:
        amplitude = perturb_fraction * (inputs_perturbed[:, i].max() - inputs_perturbed[:, i].min())
        inputs_perturbed[:, i] = inputs_perturbed[:, i] + np.random.uniform(low=-amplitude / 2, high=+amplitude / 2, size=inputs_perturbed[:, i].shape)
    # Save perturbed data such that it can be read by the (existing) test_model.get_error(.) function -> broken funcitonality with update
    if save_data:
        output_file = configs['data']['perturbed_data_file']
        np.savez(output_file, inputs_perturbed=inputs_perturbed)
    return inputs_perturbed


def get_prediction(configs, inputs, batch_size=2048):
    # Gets the prediction of the models specified in the config on the supplied inputs
    model_data_path = configs['processor']["torch_model_dict"]
    prediction = np.zeros(inputs.shape[0])
    model = SurrogateModel({'torch_model_dict': model_data_path})
    with torch.no_grad():
        i_start = 0
        i_end = batch_size
        threshold = (inputs.shape[0] - batch_size)
        while i_end <= inputs.shape[0]:
            prediction[i_start:i_end] = TorchUtils.get_numpy_from_tensor(model(TorchUtils.get_tensor_from_numpy(inputs[i_start:i_end]))).flatten()
            i_start += batch_size
            i_end += batch_size
            if i_end > threshold and i_end < inputs.shape[0]:
                i_end = inputs.shape[0]
    return prediction


def get_perturbed_rmse(configs, compare_to_measurement=False, return_error=False):
    # Gets the RMSE due to all perturbation specified in the config
    # Loops over configs['perturbation']['electrodes_sets'] and
    # over['perturbation']['perturb_fraction_sets']

    # Load config data:
    electrodes_sets = configs['perturbation']['electrodes_sets']
    perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']

    # Load unperturbed data
    inputs_unperturbed, targets_measured, info = load_data(configs)
    if compare_to_measurement:
        targets = targets_measured.flatten()
    else:
        targets = get_prediction(configs, inputs_unperturbed).flatten()

    # Preallocate array and start loop
    rmse = np.zeros((len(perturb_fraction_sets), len(electrodes_sets)))
    error = np.zeros((len(perturb_fraction_sets),len(electrodes_sets),len(targets)))
    for i in range(len(perturb_fraction_sets)):
        configs['perturbation']['perturb_fraction'] = perturb_fraction_sets[i]
        for j in range(len(electrodes_sets)):
            configs['perturbation']['electrodes'] = electrodes_sets[j]
            # Perturb data, get prediciton, get error, get rmse
            inputs_perturbed = perturb_data(configs, inputs_unperturbed)
            prediction = get_prediction(configs, inputs_perturbed)
            error[i,j,:] = prediction - targets
            rmse[i, j] = np.sqrt(np.mean(error**2))
    if return_error:
        return rmse, error
    else:
        return rmse


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


def rank_low_to_high(descriptions, values, do_plot=False, ax=None):
    # Ranks the descriptions according to the values, and potentially makes a barplot out of it.
    # Potentially plots in a specified axes
    descriptions = np.array(descriptions)
    values = np.array(values)
    ranking_indices = np.argsort(-values)  # take negative of slope to order the slope from largest (most positive -> most negative) to smallest
    ranking = descriptions[ranking_indices]
    ranked_values = values[ranking_indices]
    # Plot ranking in barplot
    if do_plot:
        if ax == None:
            plt.figure()
            ax = plt.axes()
        ax.bar(range(len(descriptions)), ranked_values)
        for j in range(len(descriptions)):
            s = np.array2string(np.array(ranking[j]))
            xy = [j, ranked_values[j]]
            ax.annotate(s, xy)
        ax.set_xlabel('Ranking')
    return ranking, ranked_values


def np_object_array_mean(obj_arr):
    # Takes the averages of the object elements of a numpy array.
    # Because if you have a numpy array of a numpy array, just using np.mean(arr, axis=0) does not work.
    float_arr = np.zeros(obj_arr.size)
    for i in range(obj_arr.size):
        float_arr[i] = obj_arr[i].mean()
    return float_arr


# %% Example c0de
if __name__ == "__main__":
    # User variables
    configs = load_configs('configs/validation/multi_perturbation_multi_electrodes_configs.json')
    electrodes_sets = configs['perturbation']['electrodes_sets']
    perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']

    # Get rmse, compare to unperturbed simulation output (not to measurement output)
    rmse, error = get_perturbed_rmse(configs, compare_to_measurement=False, return_error=True)

