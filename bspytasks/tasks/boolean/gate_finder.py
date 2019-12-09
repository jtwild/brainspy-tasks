import os
import numpy as np
import matplotlib.pyplot as plt
from bspyalgo.utils.performance import perceptron, corr_coeff
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.algorithm_manager import get_algorithm
from bspyalgo.utils.io import create_directory
from bspyproc.bspyproc import get_processor


class BooleanGateTask():

    def __init__(self, configs):
        configs = self.load_directory_configs(configs)
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.load_methods(configs['algorithm_configs'])
        self.load_task_configs(configs)
        self.validation_processor_configs = configs['validation']['processor']
        self.validation_processor = get_processor(configs['validation']['processor'])

    def load_task_configs(self, configs):
        self.show_plots = configs['show_plots']
        self.base_dir = configs['results_dir']
        self.max_attempts = configs['max_attempts']

    def load_directory_configs(self, configs):
        create_directory(configs['results_dir'], overwrite=configs['overwrite'])
        # configs['algorithm_configs']['checkpoints']['save_dir'] = os.path.join(configs['results_dir'], configs['algorithm_configs']['checkpoints']['save_dir'])
        return configs

    def load_methods(self, configs):
        if configs['algorithm'] == 'gradient_descent' and configs['processor']['platform'] == 'simulation':
            self.find_gate_core = self.find_gate_with_torch
            self.ignore_gate = self.ignore_gate_with_torch
        else:
            self.find_gate_core = self.find_gate_with_numpy
            self.ignore_gate = self.ignore_gate_with_numpy

    def find_gate(self, encoded_inputs, gate, encoded_gate, mask, threshold):
        if len(np.unique(gate)) == 1:
            print('Label ', gate, ' ignored')
            excel_results = self.ignore_gate(encoded_gate)
        else:
            attempt = 1
            print('==========================================================================================')
            print(f"Gate {gate}: ")
            while True:
                excel_results = self.find_gate_core(encoded_inputs, encoded_gate, mask)
                excel_results['found'] = excel_results['accuracy'] >= threshold

                print(f"Attempt {str(attempt)}: " + self.is_found(excel_results['found']).upper() + ", Accuracy: " + str(excel_results['accuracy']) + ", Best performance: " + str(excel_results['best_performance']))

                if excel_results['found'] or attempt >= self.max_attempts:
                    if excel_results['found']:
                        print(f'VEREDICT: PASS - Gate was found successfully in {str(attempt)} attempt(s)')
                    else:
                        print(f'VEREDICT: FAILED - Gate was NOT found in {str(attempt)} attempt(s)')
                    print('==========================================================================================')
                    self.plot_gate(excel_results, mask, show_plots=self.show_plots, save_dir=self.get_plot_dir(gate, self.is_found(excel_results['found'])))
                    break
                else:
                    attempt += 1

        excel_results['gate'] = gate

        return excel_results

    def get_plot_dir(self, gate, found):
        path = os.path.join(self.base_dir, found)
        create_directory(path, overwrite=False)
        return os.path.join(path, str(gate) + '.png')

    def find_gate_with_numpy(self, encoded_inputs, encoded_gate, mask):
        excel_results = self.optimize(encoded_inputs, encoded_gate, mask)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], encoded_gate[excel_results['mask']])
        excel_results['encoded_gate'] = encoded_gate
        return excel_results

    def find_gate_with_torch(self, encoded_inputs, encoded_gate, mask):
        encoded_gate = TorchUtils.format_tensor(encoded_gate)
        excel_results = self.optimize(encoded_inputs, encoded_gate, mask)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], TorchUtils.get_numpy_from_tensor(encoded_gate[excel_results['mask']]))
        excel_results['encoded_gate'] = encoded_gate.cpu()
        # excel_results['targets'] = excel_results
        excel_results['correlation'] = corr_coeff(excel_results['best_output'][excel_results['mask']].T, excel_results['targets'].cpu()[excel_results['mask']].T)
        return excel_results

    def ignore_gate_with_torch(self, encoded_gate):
        excel_results = self.ignore_gate_core()
        excel_results['encoded_gate'] = encoded_gate.cpu()
        return excel_results

    def ignore_gate_with_numpy(self, encoded_gate):
        excel_results = self.ignore_gate_core()
        excel_results['encoded_gate'] = encoded_gate
        return excel_results

    def ignore_gate_core(self):
        excel_results = {}
        excel_results['control_voltages'] = np.nan
        excel_results['best_output'] = np.nan
        excel_results['best_performance'] = np.nan
        excel_results['accuracy'] = np.nan
        excel_results['correlation'] = np.nan
        excel_results['found'] = True
        return excel_results

    def optimize(self, encoded_inputs, encoded_gate, mask):
        algorithm_data = self.algorithm.optimize(encoded_inputs, encoded_gate, mask=mask)
        algorithm_data.judge()
        excel_results = algorithm_data.results

        return excel_results

    def plot_gate(self, row, mask, show_plots, save_dir=None):
        plt.figure()
        plt.plot(row['best_output'][mask])
        plt.plot(row['encoded_gate'][mask])
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

    def validate_gate(self, gate, transformed_inputs, control_voltages, target, mask):
        print('==========================================================================================')
        print(f"Gate {gate} validation: ")
        y_predicted = self.validation_processor.get_output_(transformed_inputs, control_voltages)
        error = ((target - y_predicted) ** 2).mean(axis=0)[0]
        print(f'ERROR: {str(error)}')
        plot_gate_validation(y_predicted[mask], target[mask], show_plots=self.show_plots, save_dir=self.get_plot_dir(gate, 'validation'))
        print('==========================================================================================')
        return error


def find_single_gate(configs, gate, threshold, verbose=False, validate=False):
    data_manager = VCDimDataManager(configs)
    test = BooleanGateTask(configs['boolean_gate_test'])
    _, transformed_inputs, readable_targets, transformed_targets, _, mask = data_manager.get_data(4, verbose)

    excel_results = test.find_gate(transformed_inputs, readable_targets[gate], transformed_targets[gate], mask, threshold)
    if validate:
        test.validate_gate(gate, transformed_inputs, excel_results['control_voltages'], excel_results['best_output'], mask)
    return excel_results


def plot_gate_validation(output, target, show_plots, save_dir=None):
    plt.figure()
    plt.plot(output)
    plt.plot(target)
    plt.ylabel('Current (nA)')
    plt.xlabel('Time')
    if save_dir is not None:
        plt.savefig(save_dir)
    if show_plots:
        plt.show()
    plt.close()


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager

    configs = load_configs('configs/benchmark_tests/capacity/template_ga.json')
    configs = configs['capacity_test']['vc_dimension_test']
    gate = '[0 1 1 0]'
    threshold = 0.875

    find_single_gate(configs, gate, threshold, validate=True)
