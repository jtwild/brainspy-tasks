from bspyproc.bspyproc import get_processor
from bspytasks.tasks.ring.classifier import RingClassificationTask
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspyproc.utils.waveform import generate_waveform, generate_mask
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.performance import perceptron
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspytasks.tasks.ring.debugger import ArchitectureDebugger

import matplotlib.pyplot as plt
import numpy as np
import os


class RingClassifierValidator():

    def __init__(self, configs):
        self.configs = configs
        self.init_processors()
        self.init_data()
        self.debugger = ArchitectureDebugger(configs)
        self.init_dirs()

    def init_data(self):
        self.data_loader = RingDataLoader(configs)
        self.test_inputs, self.test_targets, self.test_mask = self.data_loader.generate_new_data(self.configs['algorithm_configs']['processor'], gap=self.configs['ring_data']['gap'])

    def init_processors(self):
        self.validation_processor = get_processor(self.configs['validation']['processor'])
        self.processor = get_processor(self.configs['algorithm_configs']['processor'])

    def init_dirs(self):
        self.main_dir = create_directory_timestamp(os.path.join(self.configs['results_base_dir'], 'validation'), 'validation')
        self.debugger.init_dirs(self.main_dir)
        if self.processor.configs['debug'] and self.processor.configs['architecture'] == 'device_architecture':
            self.processor.init_dirs(self.main_dir, is_main=False)
        if self.validation_processor.configs['debug'] and self.validation_processor.configs['architecture'] == 'device_architecture':
            self.validation_processor.init_dirs(self.main_dir, is_main=False)
        self.debug_plots = os.path.join(self.main_dir, 'debug', 'results')
        create_directory(self.debug_plots)

    def get_model_output(self, model):
        self.processor.load_state_dict(model.copy())
        self.processor.eval()
        if self.configs['algorithm_configs']['processor']['platform'] == 'simulation':
            inputs = TorchUtils.get_tensor_from_numpy(results['inputs'])
        else:
            inputs = results['inputs']
        model_output = self.processor.forward(inputs).detach().cpu().numpy()
        return generate_waveform(model_output[:, 0], self.configs['validation']['processor']['waveform']
                                 ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])

    def get_model_test_accuracy(self):
        model_output_test = self.processor.forward(self.test_inputs).detach().cpu().numpy()
        accuracy, _, _ = perceptron(model_output_test, self.test_targets)
        return accuracy

    def get_hardware_test_accuracy(self):
        model_output_test = self.processor.forward(self.test_inputs).detach().cpu().numpy()
        accuracy, _, _ = perceptron(model_output_test, self.test_targets)
        return accuracy

    def get_hardware_output(self, model):
        self.validation_processor.load_state_dict(model.copy())
        inputs, targets, mask = self.get_validation_inputs(results)

        return self.validation_processor.get_output_(inputs, mask)[:, 0], mask

    def validate(self, results, model):
        model_output = self.get_model_output(model)
        real_output, mask = self.get_hardware_output(model)
        self.plot_validation_results(model_output, real_output, mask, self.main_dir, self.configs['show_plots'])
        self.debugger.plot_data(use_mask=False)

    def get_validation_inputs(self, results):

        targets = generate_waveform(results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        mask = generate_mask(results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])

        inputs_1 = generate_waveform(results['inputs'][:, 0], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        inputs_2 = generate_waveform(results['inputs'][:, 1], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        inputs = np.asarray([inputs_1, inputs_2]).T

        return inputs, targets, mask

    def plot_validation_results(self, model_output, real_output, mask, save_dir=None, show_plot=False):

        error = ((model_output[mask] - real_output[mask]) ** 2).mean()
        print(f'Total Error: {error}')

        plt.figure()
        plt.title(f'Comparison between Hardware and Simulation \n (MSE: {error})', fontsize=12)
        plt.plot(model_output[mask])
        plt.plot(real_output[mask], '-.')
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')

        plt.legend(['Simulation', 'Hardware'])
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'validation_plot.eps'))
            np.savez(os.path.join(self.main_dir, 'validation_plot_data'), model_output=model_output, real_output=real_output, mask=mask)
        if show_plot:
            plt.show()
            plt.close()


def load_data(base_dir):
    model_dir = os.path.join(base_dir, 'reproducibility', 'model.pth')
    results_dir = os.path.join(base_dir, 'reproducibility', 'results.pickle')
    configs_dir = os.path.join(base_dir, 'reproducibility', 'configs.json')
    model = torch.load(model_dir)
    results = pickle.load(open(results_dir, "rb"))
    configs = load_configs(configs_dir)
    configs['results_base_dir'] = base_dir
    return model, results, configs


if __name__ == '__main__':
    import torch
    import os
    import pickle
    from bspyalgo.utils.io import load_configs

    folder_name = 'searcher_0.2mV_2020_02_26_112540'
    base_dir = 'tmp/output/ring/' + folder_name
    model, results, configs = load_data(base_dir)
    val = RingClassifierValidator(configs)
    val.validate(results, model)
