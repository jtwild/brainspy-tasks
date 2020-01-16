import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from bspyproc.utils.waveform import generate_waveform


def save_plots(results, mask, configs, run=0, show_plot=False):
    plt.figure()
    plt.plot(results['best_output'][mask])
    if configs['save_plots']:
        plt.savefig(os.path.join(configs['results_base_dir'], f"output_ring_classifier_Run_{run}"))
    plt.figure()
    plt.plot(results['performance_history'])
    if configs['save_plots']:
        plt.savefig(os.path.join(configs['results_base_dir'], f"training_profile_Run_{run}"))
    if show_plot:
        plt.show()
    plt.close('all')


def plot_gate_validation(output, target, show_plot=False, save_dir=None):
    plt.figure()
    plt.plot(output)
    plt.plot(target, '-.')
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')
    plt.title('Comparison between Processor and DNPU')
    plt.legend(['Processor', 'DNPU'])
    if save_dir is not None:
        plt.savefig(save_dir)
    if show_plot:
        plt.show()
    plt.close()


def plot1(a, b, name):
    plt.figure()
    plt.plot(a, label='device')
    plt.plot(b, label='model')
    plt.title(name)
    plt.legend()
    plt.show()
    plt.close()


def read(name, configs):
    a = np.load(os.path.join('tmp', name + '.npy'))
    b = torch.load(os.path.join('tmp', name + '.pt')).detach().cpu().numpy()
    b = generate_waveform(b, configs['validation']['processor']['waveform']
                          ['amplitude_lengths'], configs['validation']['processor']['waveform']['slope_lengths'])
    return a, b


def print_error(a, b, name):
    print('Error ' + name)
    print(((a - b) ** 2).mean())


def default_plot(name, configs):
    a, b = read(name, configs)
    print_error(a, b, name)
    plot1(a, b, name)


def plot_raw_input(configs):
    name = 'raw_input'
    a, b = read(name, configs)
    print_error(a[:, 3], b[:, 0], name)
    plot1(a[:, 3], b[:, 0], name)
    print_error(a[:, 4], b[:, 1], name)
    plot1(a[:, 4], b[:, 1], name)


def plot_data(configs):
    plot_raw_input(configs)

    default_plot('device_layer_1_output_1', configs)
    default_plot('device_layer_1_output_2', configs)

    default_plot('bn_afterclip_1_1', configs)
    default_plot('bn_afterclip_1_2', configs)

    default_plot('bn_afterbatch_1_1', configs)
    default_plot('bn_afterbatch_1_2', configs)

    default_plot('bn_aftercv_1_1', configs)
    default_plot('bn_aftercv_1_2', configs)

    default_plot('device_layer_2_output_1', configs)
    default_plot('device_layer_2_output_2', configs)

    default_plot('bn_afterclip_2_1', configs)
    default_plot('bn_afterclip_2_2', configs)

    default_plot('bn_afterbatch_2_1', configs)
    default_plot('bn_afterbatch_2_2', configs)

    default_plot('bn_aftercv_2_1', configs)
    default_plot('bn_aftercv_2_2', configs)
