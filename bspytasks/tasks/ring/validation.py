from bspyproc.bspyproc import get_processor
from bspytasks.tasks.ring.classifier import RingClassificationTask
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspyproc.utils.waveform import generate_waveform, generate_mask
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.performance import perceptron

import matplotlib.pyplot as plt
import numpy as np
import os


class RingClassifierValidator():

    def __init__(self, configs):
        self.configs = configs
        self.init_processors()
        self.init_data()
        self.init_dirs()

    def init_data(self):
        self.data_loader = RingDataLoader(configs)
        self.test_inputs, self.test_targets, self.test_mask = self.data_loader.generate_new_data(self.configs['algorithm_configs']['processor'], gap=self.configs['ring_data']['gap'])

    def init_processors(self):
        self.validation_processor = get_processor(self.configs['validation']['processor'])
        self.processor = get_processor(self.configs['algorithm_configs']['processor'])

    def init_dirs(self):
        self.main_dir = os.path.join(self.configs['results_base_dir'], 'validation')
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)

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
        self.plot_validation_results(model_output[mask], real_output[mask], self.main_dir, self.configs['show_plots'])
        np.savez(os.path.join(self.main_dir, 'validation_plot_data'), model_output=model_output, real_output=real_output, mask=mask)

    def get_validation_inputs(self, results):

        targets = generate_waveform(results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        mask = generate_mask(results['targets'], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])

        inputs_1 = generate_waveform(results['inputs'][:, 0], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        inputs_2 = generate_waveform(results['inputs'][:, 1], self.configs['validation']['processor']['waveform']['amplitude_lengths'], slope_lengths=self.configs['validation']['processor']['waveform']['slope_lengths'])
        inputs = np.asarray([inputs_1, inputs_2]).T

        return inputs, targets, mask

    def plot_validation_results(self, model_output, real_output, save_dir=None, show_plot=False):
        error = ((model_output - real_output) ** 2).mean()
        print(f'Total Error: {error}')

        fig = plt.figure()
        plt.title('Comparison between Processor and DNPU')
        fig.suptitle(f'MSE: {error}', fontsize=10)
        plt.plot(model_output)
        plt.plot(real_output, '-.')
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')

        plt.legend(['Simulation', 'Validation'])
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'validation_plot.eps'))
        if show_plot:
            plt.show()
            plt.close()


if __name__ == '__main__':
    import torch
    import pickle
    from bspyalgo.utils.io import load_configs

    validation_folder = '/tmp/output/ring/'

    model = torch.load('model.pth')
    results = pickle.load(open('results.pkl', "rb"))
    configs = load_configs('configs.json')

    val = RingClassifierValidator(configs)
    val.validate(results, model)
