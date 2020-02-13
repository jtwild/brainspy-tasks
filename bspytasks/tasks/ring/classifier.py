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
from bspyalgo.utils.performance import perceptron


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
        inputs, targets, mask = self.data_loader.get_data(processor_configs=self.configs["algorithm_configs"]["processor"])
        # inputs, targets, mask = self.data_loader.generate_data(processor_configs=self.configs["algorithm_configs"]["processor"], ring_configs=self.configs['ring_data'])
        excel_results = self.optimize(inputs, targets, mask)
        excel_results = self.add_input_data(excel_results, inputs, targets, mask)

        self.excel_results = self.process_output(excel_results)

        return self.excel_results

    def add_input_data(self, results, inputs, targets, mask):
        if type(inputs) is torch.Tensor:
            results['inputs'] = TorchUtils.get_numpy_from_tensor(inputs)
        if type(targets) is torch.Tensor:
            results['targets'] = TorchUtils.get_numpy_from_tensor(targets)
        results['mask'] = mask
        return results

    def close_test(self, run=1):
        model = {}
        model['state_dict'] = self.algorithm.processor.state_dict()
        model['info'] = self.algorithm.processor.info
        model_dir = os.path.join(self.configs['results_base_dir'], f'state_dict_Run{run}.pth')
        torch.save(model, model_dir)
        self.plotter.save_plots(self.excel_results, self.configs, show_plot=self.configs["show_plots"], run=run)
        return model_dir

    def process_output(self, excel_results):
        mask = excel_results['mask']
        best_output = excel_results['best_output'][mask]
        targets = excel_results['targets'][mask]
        targets = targets[:, np.newaxis]
        excel_results['correlation'] = corr_coeff(best_output.T, targets.T)
        if self.configs["algorithm_configs"]['hyperparameters']["loss_function"] == "fisher":
            print("Using Fisher does not allow for perceptron accuracy decision.")
            excel_results['accuracy'] = -1
        else:
            excel_results['accuracy'], _, _ = perceptron(best_output, targets)
        print(f"Accuracy: {excel_results['accuracy']}")
        return excel_results

    def get_accuracy_from_model_dir(self, model_dir):
        inputs, targets, _ = self.data_loader.get_data(processor_configs=self.configs["algorithm_configs"]["processor"])
        self.algorithm.processor.load_state_dict(torch.load(model_dir, map_location=TorchUtils.get_accelerator_type())['state_dict'])
        self.algorithm.processor.eval()
        predictions = self.algorithm.processor.forward(inputs).detach().cpu().numpy()
        self.get_accuracy(predictions, targets)

    def get_accuracy(self, predictions, targets):
        if type(targets) == torch.Tensor:
            targets = targets.cpu().numpy()
        return perceptron(predictions, targets[:, np.newaxis])[0]

    def validate_task(self, model_dir):
        validation_inputs, _, validation_mask = self.data_loader.get_data(
            processor_configs=self.configs["validation"]["processor"])
        algorithm_inputs, _, algorithm_mask = self.data_loader.get_data(
            processor_configs=self.configs["algorithm_configs"]["processor"])

        self.validation_processor.load_state_dict(torch.load(model_dir, map_location=TorchUtils.get_accelerator_type()))
        self.algorithm.processor.load_state_dict(torch.load(model_dir, map_location=TorchUtils.get_accelerator_type())['state_dict'])
        self.algorithm.processor.eval()

        print("Reading target...")
        target = self.algorithm.processor.forward(algorithm_inputs).detach().cpu().numpy()
        print("Reading validation...")
        output = self.validation_processor.get_output_(validation_inputs, validation_mask)[:, 0]

        target = generate_waveform(target[:, 0], self.configs['validation']['processor']['waveform']
                                   ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'target_algorithm'), target)
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'target_algorithm_mask'), validation_mask)
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'validation_output'), output)
        np.save(os.path.join(os.path.join('tmp', 'architecture_debug'), 'validation_output_mask'), validation_mask)


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
