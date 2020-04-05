
"""
authors: H. C. Ruiz and Unai Alegre-Ibarra
"""

import numpy as np
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.input import get_map_to_voltage_vars
from bspyproc.utils.electrodes import load_voltage_ranges


def ring(sample_no, inner_radius=0.25, gap=0.5, outer_radius=1, scale=0.9, offset=-0.3):
    '''Generates labelled TorchUtilsdata of a ring with class 1 and the center with class 0
    '''
    assert outer_radius <= 1
    # scale, offset = get_map_to_voltage_vars(min_input_volt[input_indices], max_input_volt[input_indices])
    # if outer_radius:
    #     outer_radius = inner_radius + gap + outer_radius
    # else:
    #     gamma = gap / inner_radius
    #     outer_radius = inner_radius * np.sqrt(2 * (1 + gamma) + gamma**2)

    samples = (-1 * outer_radius) + (2 * outer_radius * np.random.rand(sample_no, 2))
    norm = np.sqrt(np.sum((samples)**2, axis=1))

    # Filter out samples outside the classes
    labels = np.empty(samples.shape[0])
    labels[norm < inner_radius] = 0
    labels[(norm < outer_radius) * (norm > inner_radius + gap)] = 1
    labels[norm > outer_radius] = np.nan
    labels[(norm > inner_radius) * (norm < inner_radius + gap)] = np.nan

    return samples, labels


def subsample(class0, class1):
    # Subsample the largest class
    nr_samples = min(len(class0), len(class1))
    max_array = max(len(class0), len(class1))
    indices = np.random.permutation(max_array)[:nr_samples]
    if len(class0) == max_array:
        class0 = class0[indices]
    else:
        class1 = class1[indices]
    return class0, class1


def sort(class0, class1):
    # Sort samples within each class wrt the values of input x (i.e. index 0)
    sorted_index0 = np.argsort(class0, axis=0)[:, 0]
    sorted_index1 = np.argsort(class1, axis=0)[:, 0]
    return class0[sorted_index0], class1[sorted_index1]


def filter_and_reverse(class0, class1):
    # Filter by positive and negative values of y-axis
    class0_positive_y = class0[class0[:, 1] >= 0]
    class0_negative_y = class0[class0[:, 1] < 0][::-1]
    class1_positive_y = class1[class1[:, 1] >= 0]
    class1_negative_y = class1[class1[:, 1] < 0][::-1]

    # Define input variables and their target
    class0 = np.concatenate((class0_positive_y, class0_negative_y))
    class1 = np.concatenate((class1_positive_y, class1_negative_y))
    inputs = np.concatenate((class0, class1))
    targets = np.concatenate((np.zeros_like(class0[:, 0]), np.ones_like(class1[:, 0])))

    # Reverse negative 'y' inputs for negative cases
    return inputs, targets


def process_dataset(class0, class1):
    class0, class1 = subsample(class0, class1)
    class0, class1 = sort(class0, class1)
    return filter_and_reverse(class0, class1)


# The gap needs to be in a scale from -1 to 1. This function enables to transform the gap in volts to this scale.
def transform_gap(gap_in_volts, scale):
    assert (len(scale[scale == scale.mean()]) == len(scale)), "The GAP information is going to be inaccurate because the selected input electrodes have a different voltage range. In order for this data to be accurate, please make sure that the input electrodes have the same voltage ranges."
    if len(scale) > 1:
        scale = scale[0]

    return (gap_in_volts / scale)


def generate_data(configs, sample_no, gap):
    # Get information from the electrode ranges in order to calculate the linear transformation parameters for the inputs
    min_voltage, max_voltage = load_voltage_ranges(configs)
    i = configs['input_indices']
    scale, offset = get_map_to_voltage_vars(min_voltage[i], max_voltage[i])
    gap = transform_gap(gap, scale)

    data, labels = ring(sample_no=sample_no, gap=gap, scale=scale, offset=offset)
    data, labels = process_dataset(data[labels == 0], data[labels == 1])

    # Transform dataset to control voltage range
    samples = (data * scale) + offset
    gap = gap * scale

    print(f'The input ring dataset has a {gap}V gap.')
    print(f'There are {len(data[labels == 0]) + len(samples[labels == 1])} samples')
    return data, labels


def load_data(base_dir):
    import os
    import torch
    import pickle
    from bspyalgo.utils.io import load_configs

    model_dir = os.path.join(base_dir, 'reproducibility', 'model.pt')
    results_dir = os.path.join(base_dir, 'reproducibility', 'results.pickle')
    configs_dir = os.path.join(base_dir, 'reproducibility', 'configs.json')
    model = torch.load(model_dir, map_location=TorchUtils.get_accelerator_type())
    results = pickle.load(open(results_dir, "rb"))
    configs = load_configs(configs_dir)
    configs['results_base_dir'] = base_dir
    return model, results, configs
