'''
This is a template for evolving the NN based on the boolean_logic experiment.
The only difference to the measurement scripts are on lines where the device is called.

'''
import torch
import numpy as np
import os
from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.bspyproc import get_processor
from matplotlib import pyplot as plt
from bspyalgo.utils.performance import accuracy, corr_coeff
from bspyalgo.utils.io import create_directory, create_directory_timestamp, save
from bspyproc.utils.pytorch import TorchUtils


class RingClassificationTask():

    def __init__(self, configs, is_main=True):
        self.configs = configs
        if is_main:
            self.init_dirs(configs['results_base_dir'], is_main)

    def init_dirs(self, base_dir, is_main=False):
        main_dir = 'ring_classification'
        reproducibility_dir = 'reproducibility'
        results_dir = 'results'
        if is_main:
            base_dir = create_directory_timestamp(base_dir, main_dir)
        self.reproducibility_dir = os.path.join(base_dir, reproducibility_dir)
        create_directory(self.reproducibility_dir)
        self.configs['algorithm_configs']['results_base_dir'] = base_dir
        self.algorithm = get_algorithm(self.configs['algorithm_configs'])
        self.results_dir = os.path.join(base_dir, results_dir)
        create_directory(self.results_dir)

    def reset(self):
        self.algorithm.reset()

    def run_task(self, inputs, targets, mask, save_data=False):
        algorithm_data = self.algorithm.optimize(inputs, targets, mask=mask, save_data=save_data)
        return self.judge(algorithm_data)

    def save_reproducibility_data(self, result):
        save(mode='configs', file_path=os.path.join(self.reproducibility_dir, 'configs.json'), data=self.configs)
        save(mode='torch', file_path=os.path.join(self.reproducibility_dir, 'model.pt'), data=self.algorithm.processor)
        save(mode='pickle', file_path=os.path.join(self.reproducibility_dir, 'results.pickle'), data=result)

    def judge(self, algorithm_data):

        algorithm_data.judge()
        results = algorithm_data.get_results_as_numpy()
        results = self.get_accuracy(results)
        results = self.get_correlation(results)
        return results

    def get_correlation(self, results):
        mask = results['mask']
        results['correlation'] = corr_coeff(results['best_output'][mask].T, results['targets'][mask][:, np.newaxis].T)
        return results

    def get_accuracy(self, results):
        mask = results['mask']
        print('Calculating Accuracy ... ')
        results['accuracy'] = accuracy(results['best_output'][mask],
                                       results['targets'][mask],
                                       plot=os.path.join(self.results_dir, f"perceptron.eps"))
        print(f"Accuracy: {results['accuracy']}")
        return results

    def plot_results(self, results):
        plt.figure()
        plt.plot(results['best_output'][results['mask']])
        plt.title(f"Output (nA) \n Performance: {results['best_performance'][0]} \n Accuracy: {results['accuracy']}", fontsize=12)
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.results_dir, f"output.eps"))
        plt.figure()
        plt.title(f'Learning profile', fontsize=12)
        plt.plot(results['performance_history'])
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.results_dir, f"training_profile.eps"))

        plt.figure()
        plt.title(f"Inputs (V) \n {self.configs['ring_data']['gap']}mV gap", fontsize=12)
        if type(results['inputs']) is torch.Tensor:
            inputs = results['inputs'].cpu().numpy()
            targets = results['targets'].cpu().numpy()
        else:
            inputs = results['inputs']
            targets = results['targets']
        plt.scatter(inputs[results['mask']][:, 0], inputs[results['mask']][:, 1], c=targets[results['mask']])
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.results_dir, f"input.eps"))

        if self.configs['show_plots']:
            plt.show()
        plt.close('all')


if __name__ == '__main__':
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs
    from bspytasks.tasks.ring.data_loader import RingDataLoader

    gap = 0.2
    configs = load_configs('configs/tasks/ring/template_ann_gd.json')
    configs['ring_data']['gap'] = gap
    task = RingClassificationTask(configs)
    data_loader = RingDataLoader(configs)
    inputs, targets, mask = data_loader.generate_new_data(configs['algorithm_configs']['processor'], gap=gap)
    if type(inputs) is np.ndarray:
        inputs = TorchUtils.get_tensor_from_numpy(inputs)
        targets = TorchUtils.get_tensor_from_numpy(targets)
    result = task.run_task(inputs, targets, mask)
    task.save_reproducibility_data(result)
    task.plot_results(result)
