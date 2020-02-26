import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from bspyproc.utils.waveform import generate_waveform
from bspyalgo.utils.io import create_directory


class ArchitectureDebugger():

    def __init__(self, configs):
        self.configs = configs
        self.plot_names = self.generate_plot_names(2)

    def init_dirs(self, base_dir):
        debug_path = os.path.join(base_dir, 'debug')
        self.debug_path_hardware = os.path.join(debug_path, 'hardware')
        self.debug_path_simulation = os.path.join(debug_path, 'simulation')
        self.results_path = os.path.join(debug_path, 'plots')
        create_directory(self.results_path)

    def generate_plot_names(self, layer_no):
        result = []
        for i in range(layer_no):
            result.append('device_layer_' + str(i + 1) + '_output_0')
            result.append('device_layer_' + str(i + 1) + '_output_1')

            result.append('bn_afterclip_' + str(i + 1) + '_0')
            result.append('bn_afterclip_' + str(i + 1) + '_1')

            result.append('bn_afterbatch_' + str(i + 1) + '_0')
            result.append('bn_afterbatch_' + str(i + 1) + '_1')

            result.append('bn_aftercv_' + str(i + 1) + '_0')
            result.append('bn_aftercv_' + str(i + 1) + '_1')

        return result

    def plot_comparison(self, a, b, name):
        plt.figure()
        plt.plot(a, label='device')
        plt.plot(b, label='model')
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(self.results_path, name + '.eps'))
        plt.show()
        plt.close()

    def plot_error(self, x, name):
        plt.figure()
        plt.plot(x, label='error')
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(self.results_path, name + '_error.eps'))
        plt.show()
        plt.close()

    def read(self, name, use_mask):
        a = np.load(os.path.join(self.debug_path_hardware, name + '.npy'))
        b = torch.load(os.path.join(self.debug_path_simulation, name + '.pt')).detach().cpu().numpy()
        b = generate_waveform(b, self.configs['validation']['processor']['waveform']
                              ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])
        if use_mask:
            a = a[self.a_mask]
            b = b[self.b_mask]

        return a, b

    def print_error(self, a, b, name):
        print('Error ' + name)
        error = (a - b)
        print(f'     max error: {np.max(error)}')
        print(f'     mean error: {(error ** 2).mean()}')
        print(f'     std error: {(error ** 2).std()}')
        return error

    def default_plot(self, name, use_mask):
        a, b = self.read(name, use_mask)
        error = self.print_error(a, b, name)
        self.plot_comparison(a, b, name)
        self.plot_error(error, name)

    def plot_raw_input(self, use_mask):
        name = 'raw_input'
        a, b = self.read(name, use_mask)
        self.print_error(a[:, 3], b[:, 0], name)
        self.plot_comparison(a[:, 3], b[:, 0], name)
        self.print_error(a[:, 4], b[:, 1], name)
        self.plot_comparison(a[:, 4], b[:, 1], name)

    def plot_final_result(self, use_mask=False):
        # self.a_output = np.load(os.path.join(self.debug_path_hardware, 'validation_output.npy'))
        self.a_mask = np.load(os.path.join(self.debug_path_simulation, 'validation_output_mask.npy'))
        # self.b_output = np.load(os.path.join(self.debug_path_hardware, 'target_algorithm.npy'))
        self.b_mask = np.load(os.path.join(self.debug_path_simulation, 'target_algorithm_mask.npy'))
        # if use_mask:
        #     error = ((elf.b_output[self.b_mask] - self.a_output[self.a_mask]) ** 2).mean()
        #     print(f'Total Error: {error}')

        #     self.plot_gate_validation(self.b_output[self.b_mask], self.a_output[self.a_mask], True, save_dir=os.path.join(
        #         self.configs['results_base_dir'], 'validation.eps'))
        # else:
        #     error = ((self.b_output - self.a_output) ** 2).mean()
        #     print(f'Total Error: {error}')

        #     self.plot_gate_validation(self.b_output, self.a_output, True, save_dir=os.path.join(
        #         self.configs['results_base_dir'], 'validation.eps'))

    def plot_data(self, use_mask=False):
        # self.plot_final_result(use_mask)
        self.plot_raw_input(use_mask)

        for name in self.plot_names:
            self.default_plot(name, use_mask)
        # self.plot_final_result(use_mask)
