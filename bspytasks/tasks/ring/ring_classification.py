'''
This is a template for evolving the NN based on the boolean_logic experiment.
The only difference to the measurement scripts are on lines where the device is called.

'''
import numpy as np

from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.utils.waveform import generate_waveform, generate_mask
from matplotlib import pyplot as plt
from bspytasks.utils.excel import ExcelFile
from bspyalgo.utils.performance import perceptron
from bspyalgo.utils.io import load_configs, save
# import ring_evolve as re
# import config_ring as config


class RingClassificationTask():

    def __init__(self, configs):
        self.configs = configs
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.excel_file = ExcelFile(configs['results_base_dir'] + 'capacity_test_results.xlsx')
        self.algorithm = get_algorithm(configs['algorithm_configs'])

    def init_excel_file(self, readable_targets):
        column_names = ['label', 'found', 'accuracy', 'best_output', 'control_voltages', 'correlation', 'best_performance', 'encoded_label']
        self.excel_file.init_data(readable_targets, column_names)
        self.excel_file.reset()

    def get_ring_data_from_npz(self):
        with np.load(self.configs['ring_data_path']) as data:
            inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            print('Input shape: ', inputs.shape)
            targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)
        return self.process_npz_targets(inputs, targets)

    def process_npz_targets(self, inputs, targets):
        mask0 = (targets == 0)
        mask1 = (targets == 1)
        targets[mask0] = 1
        targets[mask1] = 0
        amplitude_lengths = self.configs['algorithm_configs']['processor']['waveform']['amplitude_lengths']
        slope_lengths = self.configs['algorithm_configs']['processor']['waveform']['slope_lengths']
        mask = generate_mask(targets, amplitude_lengths, slope_lengths=slope_lengths)
        inputs = self.generate_data_waveform(inputs, amplitude_lengths, slope_lengths)
        targets = np.asarray(generate_waveform(targets, amplitude_lengths, slope_lengths)).T

        return inputs, targets, mask

    def generate_data_waveform(self, data, amplitude_lengths, slope_lengths):
        data_waveform = np.array([])
        for i in range(data.shape[1]):
            d_waveform = generate_waveform(data[:, i], amplitude_lengths, slope_lengths=slope_lengths)
            if data_waveform.shape == (0,):
                data_waveform = np.concatenate((data_waveform, d_waveform))
            else:
                data_waveform = np.vstack((data_waveform, d_waveform))
        # if len(inputs_waveform.shape) == 1:
        #    inputs_waveform = inputs_waveform[:, np.newaxis]
        return data_waveform.T  # device_model --> (samples,dimension) ; device --> (dimensions,samples)

    def save_plot(self, inputs, targets, wave_length):
        plt.figure()
        plt.plot(inputs.T)
        plt.plot(targets, 'k')
        plt.show()

    def optimize(self, inputs, targets, mask):
        algorithm_data = self.algorithm.optimize(inputs, targets, mask=mask)
        algorithm_data.judge()

        return algorithm_data.results

    def run_task(self):
        save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)
        inputs, targets, mask = self.get_ring_data_from_npz()

        excel_results = self.optimize(inputs, targets, mask)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][mask], targets[mask])


if __name__ == '__main__':
    task = RingClassificationTask(load_configs('configs/tasks/ring/template_ga.json'))
    task.run_task()
