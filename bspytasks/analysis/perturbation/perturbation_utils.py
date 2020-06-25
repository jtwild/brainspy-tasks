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
from bspysmg.measurement.data.input.input_mgr import sawtooth_wave # for generating input data
# %% Below are all the functions used for this purpose

def load_data(configs, steps=1):
    # Either loads measurement data from an npz file, or generates input data based on config file:
    if configs["data"]["type"] == 'generation':
        # For generation, sawtooth hardcoded for now. Cannot use the SMG loader because it requires a lot of configs irrelevant for us (would make a mess):
        time_points = np.arange(configs["data"]["generation_info"]["time_min"], configs["data"]["generation_info"]["time_max"], configs["data"]["generation_info"]["time_step"])
        frequency= np.array(configs["data"]["generation_info"]["frequency"])[:, np.newaxis] * configs["data"]["generation_info"]["factor"]
        phase = np.array(configs["data"]["generation_info"]["phase"])[:, np.newaxis]
        amplitude, offset = load_amplitude_offset(configs['processor']["torch_model_dict"])
        inputs = sawtooth_wave(time_points, frequency, phase, amplitude, offset).T # transpose because first dimension should be Nsamples and second dimension should be Nelectrodes
        outputs = None
        info = None
    elif configs["data"]["type"] == 'random':
        # random grid exploration
        amplitude, offset = load_amplitude_offset(configs['processor']["torch_model_dict"])
        n_samples = int(configs["data"]["random_info"]["n_samples"])
        n_electrodes = amplitude.size # better to take directly from processor, but okay
        inputs = (offset + amplitude * np.random.uniform(low=-1, high=1, size=(n_electrodes, n_samples))).T
        outputs = None
        info = None
    elif configs["data"]["type"] == 'measurement':
        # For measurement:
        # Load input data
        input_file = configs["data"]['input_data_file']
        inputs, outputs, info = test_model.load_data(input_file, steps)
    else:
        raise NotImplementedError('Input data type (from configs) not implemented.')
    return inputs, None, None

def load_amplitude_offset(model_data_path):
    # note: amplitude is half of the peak-peak value. So, amp = 0.5 * (Vmax - Vmin)
    model = SurrogateModel({'torch_model_dict': model_data_path})
    return model.amplitude.numpy()[:, np.newaxis], model.offset.numpy()[:, np.newaxis]

def perturb_data(configs, inputs_unperturbed, save_data=False, steps=1):
    # Adds noise to a specific electrode of a set of input data
    # Noise gets added to electrodes in configs['perturbation']['electrodes']
    # with an peak-peak amplitude of configs['perturbation']['perturb_fraction'] *100% of
    # that electrodes current value range

    # Load config data
    electrodes = configs['perturbation']['electrodes']
    perturb_fraction = configs['perturbation']['perturb_fraction']
    mode = configs['perturbation']['mode'] # can be absolute or relative. Relative uses perturb fraction as a relative fraction of total range of that electrode. ABsolute uses perturb_fraction as an absolute value in volts that is used for

    # Perturb the data of the required electrodes
    inputs_perturbed = inputs_unperturbed.copy() # by default, for all electrodes, the inputs are unperturbed
    for i in electrodes: # and only some electrodes get perturbed
        if  mode == 'relative':
            amplitude = perturb_fraction * (inputs_perturbed[:, i].max() - inputs_perturbed[:, i].min())
        elif mode == 'absolute':
            amplitude = perturb_fraction
        # And add perturbation to unpertubed
        perturbation = np.random.uniform(low=-amplitude, high=+amplitude, size=inputs_perturbed[:, i].shape)
        inputs_perturbed[:, i] = inputs_perturbed[:, i] + perturbation

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
        i_end = min([batch_size, inputs.shape[0]]) # if we have less inputs than batch size, i_end should be smaller.
        threshold = (inputs.shape[0] - batch_size)
        while i_end <= inputs.shape[0]:
            prediction[i_start:i_end] = TorchUtils.get_numpy_from_tensor(model(TorchUtils.get_tensor_from_numpy(inputs[i_start:i_end]))).flatten()
            i_start += batch_size
            i_end += batch_size
            if i_end > threshold and i_end < inputs.shape[0]:
                i_end = inputs.shape[0]
    return prediction


def get_perturbed_rmse(configs, compare_to_measurement=False):
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
            rmse[i, j] = np.sqrt(np.mean(error[i,j,:]**2))

    #return error compatibility broken in update in favour of also returning the inputs
    return rmse, error, inputs_unperturbed, inputs_perturbed, targets, prediction

# sort errors by input voltage
def sort_by_input_voltage(inputs, values, min_val = None, max_val = None, num_ranges = 10):
    # Sort the values by ranges in inputs. Inputs gets grouped, and indices specifying to
    # a specific group are taken together from the values array
    # inputs should be shape (X,), values should be (X,),
    # Output-wise: values_subsets will be an array of arrays, grid gives the centrepoints of ordering
    # and ranges gives the edges which are used for ordering

    # Check the inputs
    assert inputs.ndim == 1, 'Flattened inputs array expected!'
    assert values.ndim == 1, 'Flattened values array expected!'
    assert isinstance(num_ranges, int), 'Integer number of ranges expected!'
    if min_val == None:
        # If no values supplied:
        min_val = inputs.min()
    else:
        assert isinstance(min_val, (int, float)), 'Numerical input expected for min_val!'
    if max_val == None:
        max_val = inputs.max()
    else:
        assert isinstance(min_val, (int, float)), 'Numerical input expected for max_val!'

    # Start the ordering
    ranges = np.linspace(min_val, max_val, num = num_ranges+1)
    grid = (ranges[1:] + ranges[:-1]) / 2 # centrepoint of ranges
    values_subsets = np.zeros(len(grid), dtype=object)
    for i in range(len(grid)):
        extraction_condition = np.logical_and(inputs >= ranges[i], inputs < ranges[i + 1])
#        values_subsets[i] = np.extract(extraction_condition, values)
        values_subsets[i] = values[extraction_condition]
    return values_subsets, grid, ranges
# Old version below
#def sort_by_input_voltage(inputs, values, min_val=None, max_val=None, granularity=None, num_levels = 10):
#    # Sort the values by ranges in inputs. Inputs gets grouped, and indices specifying to
#    # a specific group are taken together from the values array
#    # inputs should be shape (X,), values should be (X,),
#    # Output-wise: values_subsets will be an array of arrays, grid gives the centrepoints of ordering
#    # and ranges gives the edges which are used for ordering
#    if [min_val, max_val, granularity] == [None, None, None]:
#        # If no values supplied:
#        min_val = inputs.min()
#        max_val = inputs.max()
#        granularity = (max_val - min_val) / num_levels  # by default, group in 5 groups
#    grid = np.arange(min_val, max_val, granularity)
#    ranges = np.concatenate(([min_val], (grid[1:] + grid[:-1]) / 2, [max_val]))
#    values_subsets = [[]] * len(grid)  # no preallocation because size is unknown
#    for i in range(len(grid)):
#        extraction_condition = np.logical_and(inputs >= ranges[i], inputs < ranges[i + 1])
#        values_subsets[i] = np.extract(extraction_condition, values)
#    return np.array(values_subsets), grid, ranges


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


def rank_low_to_high(descriptions, values, plot_type=None, ax=None, x_data = []):
    # Ranks the descriptions according to the values, and potentially makes a barplot out of it.
    # Potentially plots in a specified axes
    descriptions = np.array(descriptions)
    values = np.array(values)
    ranking_indices = np.argsort(-values)  # take negative of value to order the values from largest (most positive -> most negative) to smallest
    ranked_descriptions = descriptions[ranking_indices]
    ranked_values = values[ranking_indices]
    # Plot ranking in barplot
    if plot_type!=None:
        if ax == None:
            plt.figure()
            ax = plt.axes()
        if len(x_data) == 0:
            x_data = np.arange(len(descriptions))
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
    return ranked_descriptions, ranked_values, ranking_indices


def np_object_array_mean(obj_arr, nan_val = 0):
    # Takes the averages of the object elements of a numpy array.
    # Because if you have a numpy array of a numpy array, just using np.mean(arr, axis=0) does not work.
    float_arr = np.zeros(obj_arr.size)
    for i in range(obj_arr.size):
        float_arr[i] = obj_arr[i].mean()
        if np.isnan(float_arr[i]):
            float_arr[i] = nan_val
    return float_arr


# %% Example c0de
if __name__ == "__main__":
    # User variables
#    configs = load_configs('configs/analysis/perturbation/multi_perturbation_multi_electrodes_configs.json')
    configs = load_configs('configs/analysis/perturbation/single_perturbation_all_electrodes_configs.json')

    # Get rmse, compare to unperturbed simulation output (not to measurement output)
    rmse, error = get_perturbed_rmse(configs, compare_to_measurement=False, return_error=True)
    print(rmse)

