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
from bspyalgo.utils.io import save
from bspyproc.utils.pytorch import TorchUtils

from bspyalgo.utils.performance import perceptron


class RingClassificationTask():

    def __init__(self, configs):
        self.configs = configs
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.configs['results_base_dir'] = save(mode='configs', path=self.configs['results_base_dir'], filename='ring_classification_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)
        self.algorithm = get_algorithm(configs['algorithm_configs'])

    def reset(self):
        self.algorithm.reset()

    def run_task(self, inputs, targets, mask):
        algorithm_data = self.algorithm.optimize(inputs, targets, mask=mask)
        return self.judge(algorithm_data)

    def close_test(self):
        model = {}
        model['state_dict'] = self.algorithm.processor.state_dict()
        model['info'] = self.algorithm.processor.info
        model_dir = os.path.join(os.path.join(self.configs["results_base_dir"], 'reproducibility'), f"model.pth")
        torch.save(model, model_dir)
        return model_dir

    def judge(self, algorithm_data):
        algorithm_data.judge()
        results = algorithm_data.results

        mask = results['mask']
        if type(results['best_output']) is torch.Tensor:
            best_output = results['best_output'][mask].detach().cpu().numpy()
        else:
            best_output = results['best_output'][mask]
        if type(results['targets']) is torch.Tensor:
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


if __name__ == '__main__':
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs

    # configs = load_configs('configs/tasks/ring/template_gd_architecture_cdaq_to_nidaq_validation2.json')
    configs = load_configs('configs/tasks/ring/template_gd_architecture_3.json')
    task = RingClassificationTask(configs)
    result = task.run_task()
    task.close_test()
    # task.validate_task(model_dir)
    # accuracy = task.get_accuracy_from_model_dir('state_dict_Run382.pth')
    # print(f'Accuracy: {accuracy}')
    # task.validate_task('state_dict_Run382.pth')
    # plotter = ArchitecturePlotter(configs)

    # print('PLOTTING DATA WITH MASK')
    # plotter.plot_data(use_mask=True)
    # print('PLOTTING DATA WITHOUT MASK')
    # plotter.plot_final_result()
    # plotter.plot_data()
