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


def plot_data(configs):
    import bspyproc.utils.waveform as wf

    # name = 'raw_input'
    # a, b = read(name, configs)
    # print_error(a[:,3],b[:,0], name)
    # plot1(a[:,3],b[:,0], name)
    # print_error(a[:,4],b[:,1], name)
    # plot1(a[:,4],b[:,1], name)

    # name = 'layer_1_output_1'
    # a, b = read(name, configs)
    # print_error(a[:,0],b[:,0], name)
    # plot1(a[:,0],b[:,0], name)

    # name = 'layer_1_output_2'
    # a, b = read(name, configs)
    # print_error(a[:,0],b[:,0], name)
    # plot1(a[:,0],b[:,0], name)

    # name = 'bn_afterclip_1_1'
    # a, b = read(name, configs)
    # # print_error(a[:,0],b[:,0], name)
    # plot1(a[:,0],b[:,0], name)

    # name = 'bn_afterclip_1_2'
    # a, b = read(name, configs)
    # # print_error(a[:,0],b[:,0], name)
    # plot1(a[:,0],b[:,0], name)

    # name = 'bn_afterbatch_1'
    # a = np.load('bn_afterbatch_1.npy')
    # b = torch.load('bn_afterbatch_1.pt').detach().cpu().numpy()
    # b = generate_waveform(b, configs['validation']['processor']['waveform']['amplitude_lengths'], 0)
    # c = np.load('bn_afterbatch_2.npy')
    # # print_error(a,b[:,0], name)
    # plot1(a,b[:,0], name)

    # print_error(c,b[:,1], name)
    # plot1(c,b[:,1], name)

    # name = 'bn_aftercv_1'
    # a, b = read(name, configs)
    # # print_error(a,b[:,0], name)
    # plot1(a,b[:,0], name)

    # name = 'bn_aftercv_2'
    # a, b = read(name, configs)
    # # print_error(a,b[:,1], name)
    # plot1(a,b[:,1], name)

    l1_np = np.load('layer_1_output_processed.npy')
    l1_tr = torch.load('layer_1_output_processed.pt').detach().cpu().numpy()
    l1_tr = generate_waveform(l1_tr, configs['validation']['processor']['waveform']
                              ['amplitude_lengths'], 20)

    # print('Error')
    # print(((l1_np[:, 0] - l1_tr[:, 0]) ** 2).mean())

    plt.plot(l1_np[:, 14 + 3])
    plt.plot(l1_tr[:, 0])
    plt.show()

    plt.plot(l1_np[:, 14 + 4])
    plt.plot(l1_tr[:, 1])
    plt.show()

    l2_1_np = np.load('layer_2_output_2.npy')
    l2_1_tr = torch.load('layer_2_output_2.pt').detach().cpu().numpy()
    l2_1_tr = generate_waveform(l2_1_tr, configs['validation']['processor']['waveform']
                                ['amplitude_lengths'], 20)
    # print('Error')
    # print(((l2_1_np[:, 0] - l2_1_tr[:, 0]) ** 2).mean())

    plt.plot(l2_1_np[:, 0])
    plt.plot(l2_1_tr[:, 0])
    plt.show()

    l2_2_np = np.load('layer_2_output_2.npy')
    l2_2_tr = torch.load('layer_2_output_2.pt').detach().cpu().numpy()
    l2_2_tr = generate_waveform(l2_2_tr, configs['validation']['processor']['waveform']
                                ['amplitude_lengths'], 20)

    # print('Error')
    # print(((l2_2_np[:, 0] - l2_2_tr[:, 0]) ** 2).mean())

    plt.plot(l2_2_np[:, 0])
    plt.plot(l2_2_tr[:, 0])
    plt.show()

    l2_np = np.load('layer_2_output_processed.npy')
    l2_tr = torch.load('layer_2_output_processed.pt').detach().cpu().numpy()
    l2_2_tr = generate_waveform(b, configs['validation']['processor']['waveform']
                                ['amplitude_lengths'], 0)

    print('Error')
    print(((l2_np[:, 0] - l2_tr[:, 0]) ** 2).mean())

    plt.plot(l2_np[:, 28 + 3])
    plt.plot(l2_tr[:, 0])
    plt.show()

    plt.plot(l2_np[:, 28 + 4])
    plt.plot(l2_tr[:, 1])
    plt.show()


def print_error(a, b, name):
    print('Error ' + name)
    print(((a - b) ** 2).mean())
