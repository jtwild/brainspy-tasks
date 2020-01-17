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
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspyproc.utils.waveform import generate_waveform
from bspytasks.tasks.ring.plotter import ArchitecturePlotter


class RingClassificationTask():

    def __init__(self, configs):
        self.configs = configs
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.configs['results_base_dir'] = save(mode='configs', path=self.configs['results_base_dir'], filename='ring_classification_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)
        self.data_loader = RingDataLoader(configs)
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.plotter = ArchitecturePlotter(configs)
        if 'validation' in configs:
            self.validation_processor = get_processor(configs['validation']['processor'])

    def optimize(self, inputs, targets, mask):
        algorithm_data = self.algorithm.optimize(inputs, targets, mask=mask)
        algorithm_data.judge()

        return algorithm_data.results

    def run_task(self, run=1):
        inputs, targets, mask = self.data_loader.get_ring_data_from_npz(processor_configs=self.configs["algorithm_configs"]["processor"])
        excel_results = self.optimize(inputs, targets, mask)
        excel_results = self.process_output(excel_results, targets, mask)
        torch.save(self.algorithm.processor.state_dict(), os.path.join(self.configs['results_base_dir'], f'state_dict_Run{run}.pth'))
        self.plotter.save_plots(excel_results, mask, self.configs, show_plot=self.configs["show_plots"], run=run)

        return excel_results

    def process_output(self, excel_results, targets, mask):
        best_output = excel_results['best_output'][mask]
        targets = targets[mask].cpu().numpy()
        targets = targets[:, np.newaxis]
        excel_results['correlation'] = corr_coeff(best_output.T, targets.T)
        if self.configs["algorithm_configs"]['hyperparameters']["loss_function"] == "fisher":
            print("Using Fisher does not allow for perceptron accuracy decision.")
        else:
            excel_results['accuracy'], _, _ = perceptron(best_output, targets)
        return excel_results

    def validate_task(self):
        validation_inputs, _, validation_mask = self.data_loader.get_ring_data_from_npz(
            processor_configs=self.configs["validation"]["processor"])
        algorithm_inputs, _, algorithm_mask = self.data_loader.get_ring_data_from_npz(
            processor_configs=self.configs["algorithm_configs"]["processor"])

        self.validation_processor.load_state_dict(torch.load('state_dict_Run33.pth', map_location=TorchUtils.get_accelerator_type()))
        self.algorithm.processor.load_state_dict(torch.load('state_dict_Run33.pth', map_location=TorchUtils.get_accelerator_type()))
        self.algorithm.processor.eval()

        print("Reading target...")
        target = self.algorithm.processor.forward(algorithm_inputs).detach().cpu().numpy()
        print("Reading validation...")
        output = self.validation_processor.get_output_(validation_inputs, validation_mask)

        target = generate_waveform(target[:, 0], self.configs['validation']['processor']['waveform']
                                   ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'target_algorithm'), target)
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'target_algorithm_mask'), algorithm_mask)
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'validation_output'), output[:, 0])
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'validation_output_mask'), validation_mask)


if __name__ == '__main__':
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt

    from bspyalgo.utils.io import load_configs

    task = RingClassificationTask(load_configs('configs/tasks/ring/template_gd_architecture_2.json'))
    # result = task.run_task()
    # task.close_test()

    # excel = pd.read_pickle(os.path.join(task.configs["results_base_dir"], 'results.pkl'))

    task.validate_task()
    plotter = ArchitecturePlotter(load_configs('configs/tasks/ring/template_gd_architecture_2.json'))
    print('PLOTTING DATA WITH MASK')
    plotter.plot_data(use_mask=True)
    print('PLOTTING DATA WITHOUT MASK')
    plotter.plot_data()
    # plot_data(load_configs('configs/tasks/ring/template_gd_architecture_2.json'))
    # plot_data(load_configs('configs/tasks/ring/template_gd_architecture_cdaq_to_nidaq_validation2.json'))

    # configs = load_configs('configs/tasks/ring/template_gd_architecture_cdaq_to_nidaq_validation2.json')
    # rdl = RingDataLoader(configs)
    # validation_inputs, _, validation_mask = rdl.get_ring_data_from_npz(
    #     processor_configs=configs["validation"]["processor"])
    # algorithm_inputs, _, algorithm_mask = rdl.get_ring_data_from_npz(
    #     processor_configs=configs["algorithm_configs"]["processor"])
    # plot_masked_data(configs, validation_mask, algorithm_mask)
