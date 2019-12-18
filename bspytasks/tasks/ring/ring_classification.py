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
from bspyproc.utils.pytorch import TorchUtils
# import ring_evolve as re
# import config_ring as config


class RingClassificationTask():

    def __init__(self, configs):
        self.configs = configs
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.excel_file = ExcelFile(configs['results_base_dir'] + 'capacity_test_results.xlsx')
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        if 'validation' in configs:
            self.validation_processor = get_processor(configs['validation']['processor'])

    def init_excel_file(self, readable_targets):
        column_names = ['accuracy', 'best_output', 'best_performance', 'control_voltages', 'inputs', 'mask', 'performance_history', 'targets', 'bn_1_mean', 'bn_1_var', 'bn_2_mean', 'bn_2_var']
        self.excel_file.init_data(column_names)
        self.excel_file.reset()

    def get_ring_data_from_npz(self, use_torch=False):
        with np.load(self.configs['ring_data_path']) as data:
            inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            print('Input shape: ', inputs.shape)
            targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)
        return self.process_npz_targets(inputs, targets, use_torch=use_torch)

    def process_npz_targets(self, inputs, targets, use_torch=False):
        mask0 = (targets == 0)
        mask1 = (targets == 1)
        targets[mask0] = 1
        targets[mask1] = 0
        amplitude_lengths = self.configs['algorithm_configs']['processor']['waveform']['amplitude_lengths']
        slope_lengths = self.configs['algorithm_configs']['processor']['waveform']['slope_lengths']
        mask = generate_mask(targets, amplitude_lengths, slope_lengths=slope_lengths)
        inputs = self.generate_data_waveform(inputs, amplitude_lengths, slope_lengths)
        targets = np.asarray(generate_waveform(targets, amplitude_lengths, slope_lengths)).T

        if use_torch:
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

    def save_plot(self, inputs, targets, show_plot=False, save_dir=None):
        plt.figure()
        plt.plot(inputs)
        plt.plot(targets, 'k')
        if save_dir is not None:
            plt.savefig(save_dir)
        if show_plot:
            plt.show()
        plt.close()

    def optimize(self, inputs, targets, mask):
        algorithm_data = self.algorithm.optimize(inputs, targets, mask=mask)
        algorithm_data.judge()

        return algorithm_data.results

    def run_task(self):
        save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)
        inputs, targets, mask = self.get_ring_data_from_npz(use_torch=(self.configs['algorithm_configs']['processor']["simulation_type"] == 'neural_network'))
        self.init_excel_file(targets)
        excel_results = self.optimize(inputs, targets, mask)
        best_output = excel_results['best_output'][mask]
        targets = targets[mask].cpu().numpy()
        targets = targets[:, np.newaxis]
        excel_results['accuracy'], _, _ = perceptron(best_output, targets)
        bn_statistics = self.algorithm.processor.get_bn_statistics()
        for key in bn_statistics.keys():
            excel_results[key + '_mean'] = bn_statistics[key]['mean']
            excel_results[key + '_var'] = bn_statistics[key]['var']

        self.save_plot(best_output, targets, show_plot=True, save_dir=os.path.join(self.configs['results_base_dir'], 'result.png'))
        self.close_test(excel_results)
        return excel_results

    def validate_task(self, control_voltages, bn_statistics=None, use_torch=False):
        inputs, _, mask = self.get_ring_data_from_npz(use_torch=use_torch)
        slopped_plato = generate_slopped_plato(self.configs['validation']['processor']['waveform']['slope_lengths'], inputs.shape[0])[np.newaxis, :]
        control_voltages = slopped_plato * control_voltages[:, np.newaxis]
        if bn_statistics is not None:
            self.validation_processor.set_batch_normalistaion_values(bn_statistics)

        output = self.validation_processor.get_output_(inputs, control_voltages.T, mask)

        return output

    def close_test(self, results):
        self.excel_file.add_result(results)
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


if __name__ == '__main__':
    import pandas as pd
    task = RingClassificationTask(load_configs('configs/tasks/ring/template_gd_architecture.json'))
    excel = pd.read_excel('tmp/output/ring_test_results/capacity_test_results.xlsx')

    bn_statistics = load_bn_values(excel)
    control_voltages = get_numpy_from_series(excel['control_voltages'])

    task.validate_task(control_voltages, bn_statistics=bn_statistics, use_torch=False)

    # result = task.run_task()
    # print(f"Control voltages: {result['control_voltages']}")
