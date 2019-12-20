'''
This is a template for evolving the NN based on the boolean_logic experiment.
The only difference to the measurement scripts are on lines where the device is called.

'''
import torch
import numpy as np
import os
from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.bspyproc import get_processor
from bspyproc.utils.waveform import generate_waveform, generate_mask
from matplotlib import pyplot as plt
from bspytasks.utils.excel import ExcelFile
from bspyalgo.utils.performance import perceptron, corr_coeff
from bspyalgo.utils.io import load_configs, save, create_directory
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.waveform import generate_slopped_plato, generate_waveform


class RingClassificationTask():

    def __init__(self, configs):
        self.configs = configs
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.configs['results_base_dir'] = save(mode='configs', path=self.configs['results_base_dir'], filename='ring_classification_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)
        self.excel_file = ExcelFile(os.path.join(self.configs['results_base_dir'], 'experiment_results.xlsx'))
        self.init_excel_file()
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        if 'validation' in configs:
            self.validation_processor = get_processor(configs['validation']['processor'])

    def init_excel_file(self):
        column_names = ['accuracy', 'best_output', 'best_performance', 'correlation', 'control_voltages', 'inputs', 'mask', 'performance_history', 'targets']
        self.excel_file.init_data(column_names)
        self.excel_file.reset()

    def get_ring_data_from_npz(self, processor_configs):
        with np.load(self.configs['ring_data_path']) as data:
            inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            print('Input shape: ', inputs.shape)
            targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)
        return self.process_npz_targets(inputs, targets, processor_configs=processor_configs)

    def process_npz_targets(self, inputs, targets, processor_configs):
        mask0 = (targets == 0)
        mask1 = (targets == 1)
        targets[mask0] = 1
        targets[mask1] = 0
        amplitude_lengths = processor_configs['waveform']['amplitude_lengths']
        slope_lengths = processor_configs['waveform']['slope_lengths']
        mask = generate_mask(targets, amplitude_lengths, slope_lengths=slope_lengths)
        inputs = self.generate_data_waveform(inputs, amplitude_lengths, slope_lengths)
        targets = np.asarray(generate_waveform(targets, amplitude_lengths, slope_lengths)).T

        if processor_configs["simulation_type"] == 'neural_network' and processor_configs["network_type"] == 'dnpu':
            inputs = TorchUtils.get_tensor_from_numpy(inputs)
            targets = TorchUtils.get_tensor_from_numpy(targets)

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

    def save_plots(self, results, mask, run=0, show_plot=False):
        plt.figure()
        plt.plot(results['best_output'][mask])
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.configs['results_base_dir'], f"output_ring_classifier_Run_{run}"))
        plt.figure()
        plt.plot(results['performance_history'])
        if self.configs['save_plots']:
            plt.savefig(os.path.join(self.configs['results_base_dir'], f"training_profile_Run_{run}"))
        if show_plot:
            plt.show()
        plt.close('all')

    def optimize(self, inputs, targets, mask):
        algorithm_data = self.algorithm.optimize(inputs, targets, mask=mask)
        algorithm_data.judge()

        return algorithm_data.results

    def run_task(self, run=1):
        self.algorithm.reset_processor()
        inputs, targets, mask = self.get_ring_data_from_npz(processor_configs=self.configs["algorithm_configs"]["processor"])
        # self.init_excel_file(targets)
        excel_results = self.optimize(inputs, targets, mask)
        best_output = excel_results['best_output'][mask]
        targets = targets[mask].cpu().numpy()
        targets = targets[:, np.newaxis]
        excel_results['correlation'] = corr_coeff(best_output.T, targets.T)
        if self.configs["algorithm_configs"]['hyperparameters']["loss_function"] is "fisher":
            print("Using Fisher does not allow for perceptron accuracy decision.")
        else:
            excel_results['accuracy'], _, _ = perceptron(best_output, targets)

        torch.save(self.algorithm.processor.state_dict(), 'test.pth')
        # excel_results['scale'] = self.algorithm.processor.get_scale()
        # excel_results['offset'] = self.algorithm.processor.get_offset()

        # excel_results['bn_stats'] = self.algorithm.processor.get_bn_dict()  # self.set_bn_stats(excel_results)
        self.save_plots(excel_results, mask, show_plot=self.configs["show_plots"], run=run)
        self.excel_file.add_result(excel_results)
        # self.close_test(excel_results)
        return excel_results

    def set_bn_stats(self, excel_results):
        bn_statistics = self.algorithm.processor.get_bn_statistics()
        for key in bn_statistics.keys():
            excel_results[key + '_mean'] = bn_statistics[key]['mean']
            excel_results[key + '_var'] = bn_statistics[key]['var']
        return excel_results

    def validate_task(self, excel, use_torch=False):

        value = excel.iloc[excel['best_performance'].astype(float).idxmin()]
        # bn_statistics = load_bn_values(value)
        # control_voltages = value['control_voltages'].reshape(25)
        target = value['best_output']

        inputs, _, mask = self.get_ring_data_from_npz(processor_configs=self.configs["validation"]["processor"])
        slopped_plato = generate_slopped_plato(self.configs['validation']['processor']['waveform']['slope_lengths'], inputs.shape[0])[np.newaxis, :]
        # control_voltages = slopped_plato * control_voltages[:, np.newaxis]
        # if bn_statistics is not None:
        #     self.validation_processor.set_batch_normalistaion_values(bn_statistics)
        self.validation_processor.load_state_dict(torch.load('test.pth'))
        # self.validation_processor.initialise_parameters(control_voltages, )
        # self.validation_processor.set_scale_and_offset(offset=excel['offset'][0], scale=excel['scale'][0])
        # self.validation_processor.set_control_voltages(control_voltages)
        self.validation_processor.eval()
        target = generate_waveform(target[:, 0], self.configs['validation']['processor']['waveform']['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])
        # output = self.validation_processor.get_output_(inputs, control_voltages.T, mask)
        output = self.validation_processor.forward(inputs)
        output = TorchUtils.get_numpy_from_tensor(output.detach())
        error = ((target[mask] - output[mask]) ** 2).mean()
        self.plot_gate_validation(output[:, 0][mask], target[mask], self.configs['show_plots'], save_dir=os.path.join(self.configs['results_base_dir'], 'validation.png'))

        return error

    def close_test(self):
        self.excel_file.data.to_pickle(os.path.join(self.configs["results_base_dir"], 'results.pkl'))
        self.excel_file.save_tab('Ring problem')
        self.excel_file.close_file()

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
        # excel_results['correlation'] = corr_coeff(excel_results['best_output'][excel_results['mask']].T, excel_results['targets'].cpu()[excel_results['mask']].T)
        return excel_results

    def plot_gate_validation(self, output, target, show_plots, save_dir=None):
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
    import pandas as pd
    from bspytasks.utils.excel import load_bn_values

    task = RingClassificationTask(load_configs('configs/tasks/ring/template_gd_architecture.json'))
    result = task.run_task()
    task.close_test()

    excel = pd.read_pickle(os.path.join(task.configs["results_base_dir"], 'results.pkl'))

    error = task.validate_task(excel, use_torch=True)
    print(f'Error: {error}')
