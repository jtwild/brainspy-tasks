import os
import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.utils.performance import perceptron, corr_coeff
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.algorithm_manager import get_algorithm
from bspyalgo.utils.io import create_directory


class BooleanGateTask():

    def __init__(self, configs):
        configs = self.load_directory_configs(configs)
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.load_methods(configs['algorithm_configs'])
        self.load_task_configs(configs)

    def load_task_configs(self, configs):
        self.show_plots = configs['show_plots']
        self.base_dir = configs['results_dir']
        self.max_attempts = configs['max_attempts']

    def load_directory_configs(self, configs):
        create_directory(configs['results_dir'], overwrite=configs['overwrite'])
        configs['algorithm_configs']['checkpoints']['save_dir'] = os.path.join(configs['results_dir'], configs['algorithm_configs']['checkpoints']['save_dir'])
        return configs

    def load_methods(self, configs):
        if configs['algorithm'] == 'gradient_descent' and configs['processor']['platform'] == 'simulation':
            self.find_label_core = self.find_label_with_torch
            self.ignore_label = self.ignore_label_with_torch
        else:
            self.find_label_core = self.find_label_with_numpy
            self.ignore_label = self.ignore_label_with_numpy

    def find_label(self, encoded_inputs, label, encoded_label, mask, threshold):
        if len(np.unique(label)) == 1:
            print('Label ', label, ' ignored')
            excel_results = self.ignore_label(encoded_label)
        else:
            attempt = 1
            print('==========================================================================================')
            print(f"Gate {label}: ")
            while True:
                excel_results = self.find_label_core(encoded_inputs, encoded_label, mask)
                excel_results['found'] = excel_results['accuracy'] >= threshold

                print(f"Attempt {str(attempt)}: " + self.is_found(excel_results['found']).upper() + ", Accuracy: " + str(excel_results['accuracy']) + ", Best performance: " + str(excel_results['best_performance']))

                if excel_results['found'] or attempt >= self.max_attempts:
                    if excel_results['found']:
                        print(f'VEREDICT: PASS - Gate was found successfully in {str(attempt)} attempt(s)')
                    else:
                        print(f'VEREDICT: FAILED - Gate was NOT found in {str(attempt)} attempt(s)')
                    print('==========================================================================================')
                    self.plot_gate(excel_results, mask, show_plots=self.show_plots, save_dir=self.get_plot_dir(label, excel_results['found']))
                    break
                else:
                    attempt += 1

        excel_results['label'] = label

        return excel_results

    def get_plot_dir(self, label, found):
        path = os.path.join(self.base_dir, self.is_found(found))
        create_directory(path, overwrite=False)
        return os.path.join(path, str(label) + '.png')

    def find_label_with_numpy(self, encoded_inputs, encoded_label, mask):
        excel_results = self.optimize(encoded_inputs, encoded_label, mask)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], encoded_label[excel_results['mask']])
        excel_results['encoded_label'] = encoded_label
        return excel_results

    def find_label_with_torch(self, encoded_inputs, encoded_label, mask):
        encoded_label = TorchUtils.format_tensor(encoded_label)
        excel_results = self.optimize(encoded_inputs, encoded_label, mask)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], TorchUtils.get_numpy_from_tensor(encoded_label[excel_results['mask']]))
        excel_results['encoded_label'] = encoded_label.cpu()
        # excel_results['targets'] = excel_results
        excel_results['correlation'] = corr_coeff(excel_results['best_output'][excel_results['mask']].T, excel_results['targets'].cpu()[excel_results['mask']].T)
        return excel_results

    def ignore_label_with_torch(self, encoded_label):
        excel_results = self.ignore_label_core()
        excel_results['encoded_label'] = encoded_label.cpu()
        return excel_results

    def ignore_label_with_numpy(self, encoded_label):
        excel_results = self.ignore_label_core()
        excel_results['encoded_label'] = encoded_label
        return excel_results

    def ignore_label_core(self):
        excel_results = {}
        excel_results['control_voltages'] = np.nan
        excel_results['best_output'] = np.nan
        excel_results['best_performance'] = np.nan
        excel_results['accuracy'] = np.nan
        excel_results['correlation'] = np.nan
        excel_results['found'] = True
        return excel_results

    def optimize(self, encoded_inputs, encoded_label, mask):
        algorithm_data = self.algorithm.optimize(encoded_inputs, encoded_label, mask=mask)
        algorithm_data.judge()
        excel_results = algorithm_data.results

        return excel_results

    def plot_gate(self, row, mask, show_plots, save_dir=None):
        plt.figure()
        plt.plot(row['best_output'][mask])
        plt.plot(row['encoded_label'][mask])
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')
        if save_dir is not None:
            plt.savefig(save_dir)
        if show_plots:
            plt.show()
        plt.close()

    def is_found(self, found):
        if found:
            return 'found'
        else:
            return 'not_found'


def find_gate(configs, gate, threshold, verbose=False):
    data_manager = VCDimDataManager(configs)
    test = BooleanGateTask(configs['boolean_gate_test'])
    _, transformed_inputs, readable_targets, transformed_targets, _, mask = data_manager.get_data(4, verbose)

    return test.find_label(transformed_inputs, readable_targets[gate], transformed_targets[gate], mask, threshold)


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager

    configs = load_configs('configs/benchmark_tests/capacity/template_ga.json')
    configs = configs['capacity_test']['vc_dimension_test']
    gate = '[0 1 1 0]'
    threshold = 0.875

    find_gate(configs, gate, threshold)
