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
from bspyalgo.utils.performance import perceptron, corr_coeff
from bspyalgo.utils.io import create_directory, create_directory_timestamp, save
from bspyproc.utils.pytorch import TorchUtils

from bspyalgo.utils.performance import perceptron


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
        save(mode='torch', file_path=os.path.join(self.reproducibility_dir, 'model.pth'), data=self.algorithm.processor)
        save(mode='pickle', file_path=os.path.join(self.reproducibility_dir, 'results.pickle'), data=result)

    def judge(self, algorithm_data):
        algorithm_data.judge()
        results = algorithm_data.results

        mask = results['mask']
        if isinstance(results['best_output'], torch.Tensor):
            best_output = results['best_output'][mask].detach().cpu().numpy()
        else:
            best_output = results['best_output'][mask]
        if isinstance(results['targets'], torch.Tensor):
            targets = results['targets'][mask][:, np.newaxis].detach().cpu().numpy()
        else:
            targets = results['targets'][mask][:, np.newaxis]

        results = self.get_correlation(results, best_output, targets)
        return self.get_accuracy(results, best_output, targets)

    def get_correlation(self, results, best_output, targets):
        results['correlation'] = corr_coeff(best_output.T, targets.T)
        return results

    def get_accuracy(self, results, best_output, targets):
        if self.configs["algorithm_configs"]['hyperparameters']["loss_function"] == "fisher":
            print("Using Fisher does not allow for perceptron accuracy decision.")
            results['accuracy'] = -1
        else:
            results['accuracy'], _, _ = perceptron(best_output, targets)
        print(f"Accuracy: {results['accuracy']}")
        return results

    def plot_results(self, results):
        plt.figure()
        plt.plot(results['best_output'][mask])
        plt.title(f"Output (nA) [Performance: {results['best_performance'][0]}, Accuracy: {results['accuracy']}]", fontsize=12)
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.results_dir, f"output.eps"))
        plt.figure()
        plt.title(f'Learning profile')
        plt.plot(results['performance_history'])
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.results_dir, f"training_profile.eps"))

        plt.figure()
        plt.title(f"Inputs (V) with {self.configs['ring_data']['gap']}mV gap", fontsize=12)
        # if type(results['inputs']) is torch.Tensor:
        #     inputs = inputs.cpu().numpy()
        # if type(targets) is torch.Tensor:
        #     targets = targets.cpu().numpy()
        plt.scatter(results['inputs'][results['mask']][:, 0], results['inputs'][results['mask']][:, 1], c=results['targets'][results['mask']])
        # gap=inputs[targets == 0].max() - inputs[targets == 1].max()
        # print(f"Input gap is {gap} V")
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.results_dir, f"input.eps"))

        if show_plot:
            plt.show()
        plt.close('all')


if __name__ == '__main__':
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs
    from bspytasks.tasks.ring.data_loader import RingDataLoader

    gap = 0.2
    # configs = load_configs('configs/tasks/ring/template_gd_architecture_cdaq_to_nidaq_validation2.json')
    configs = load_configs('configs/tasks/ring/template_gd_architecture_3.json')
    configs['ring_data']['gap'] = gap
    task = RingClassificationTask(configs)
    data_loader = RingDataLoader(configs)
    inputs, targets, mask = data_loader.generate_new_data(configs['algorithm_configs']['processor'], gap=gap)
    result = task.run_task(inputs, targets, mask)
    task.save_reproducibility_data(result)
    task.plot_results(result, show_plot=True)
