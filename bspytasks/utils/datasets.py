
"""
authors: H. C. Ruiz and Unai Alegre-Ibarra
"""

import numpy as np
from bspyproc.utils.pytorch import TorchUtils


def ring(sample_no, inner_radius=0.1, gap=0.2, outer_radius=None):
    '''Generates labelled TorchUtilsdata of a ring with class 1 and the center with class 0
    '''
    if outer_radius:
        outer_radius = inner_radius + gap + outer_radius
    else:
        gamma = gap / inner_radius
        outer_radius = inner_radius * np.sqrt(2 * (1 + gamma) + gamma**2)

    samples = (-1 * outer_radius) + 2 * outer_radius * np.random.rand(sample_no, 2)
    norm = np.sqrt(np.sum(samples**2, axis=1))
    labels = np.empty(samples.shape[0])

    labels[norm < inner_radius] = 0
    labels[(norm < outer_radius) * (norm > inner_radius + gap)] = 1
    # Filter out samples outside the classes
    labels[norm > outer_radius] = np.nan
    labels[(norm > inner_radius) * (norm < inner_radius + gap)] = np.nan
    return samples[labels == 0], samples[labels == 1]


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


def generate_data(configs):
    class0, class1 = ring(sample_no=configs['sample_no'], gap=configs['gap'], outer_radius=0.1)
    return process_dataset(class0, class1)


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    configs = {'sample_no': 10000, 'gap': 0.4}
    waveforms, targets = generate_data(configs)

    print(f'sample efficiency: {configs["sample_no"]/len(waveforms)}')
    plt.figure()
    plt.plot(waveforms[:, 0], waveforms[:, 1], '.')
    plt.show()
