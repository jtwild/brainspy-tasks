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
        validation_inputs, _, validation_mask = self.get_ring_data_from_npz(
            processor_configs=self.configs["validation"]["processor"])
        algorithm_inputs, _, algorithm_mask = self.get_ring_data_from_npz(
            processor_configs=self.configs["algorithm_configs"]["processor"])

        self.validation_processor.load_state_dict(torch.load('test.pth'))

        target = self.algorithm.processor.forward(algorithm_inputs).detach().cpu().numpy()
        target = generate_waveform(target[algorithm_mask][:, 0], self.configs['validation']['processor']['waveform']
                                   ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])

        output = self.validation_processor.get_output_(validation_inputs, validation_mask)

        error = ((target[validation_mask] - output[validation_mask]) ** 2).mean()
        self.plot_gate_validation(output[:, 0][validation_mask], target[validation_mask], self.configs['show_plots'], save_dir=os.path.join(
            self.configs['results_base_dir'], 'validation.png'))

        return error

    def close_test(self):
        # self.excel_file.data.to_pickle(os.path.join(self.configs["results_base_dir"], 'results.pkl'))
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
        plt.plot(target, '.')
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')
        plt.title('Comparison between Processor and DNPU')
        plt.legend(['Processor', 'DNPU'])
        if save_dir is not None:
            plt.savefig(save_dir)
        if show_plots:
            plt.show()
        plt.close()


def plot_data():
    l1_1_np = np.load('layer_1_output_1.npy')
    l1_1_tr = torch.load('layer_1_output_1.pt').detach().cpu().numpy()

    print('Error')
    print(((l1_1_np[:, 0] - l1_1_tr[:, 0]) ** 2).mean())

    plt.plot(l1_1_np[:, 0])
    plt.plot(l1_1_tr[:, 0])
    plt.title('layer_1_output_1')
    plt.show()

    l1_2_np = np.load('layer_1_output_2.npy')
    l1_2_tr = torch.load('layer_1_output_2.pt').detach().cpu().numpy()

    print('Error')
    print(((l1_2_np[:, 0] - l1_2_np[:, 0]) ** 2).mean())

    plt.plot(l1_2_np[:, 0])
    plt.plot(l1_2_tr[:, 0])
    plt.title('layer_1_output_2')
    plt.show()

    bn_afterclip_1_np = np.load('bn_afterclip_1_1.npy')
    bn_afterclip_1_tr = torch.load(
        'bn_afterclip_1_1.pt').detach().cpu().numpy()

    print('Error')
    print(((bn_afterclip_1_np[:, 0] - bn_afterclip_1_tr[:, 0]) ** 2).mean())

    plt.plot(bn_afterclip_1_np[:, 0])
    plt.plot(bn_afterclip_1_tr[:, 0])
    plt.title('bn_afterclip_1_1')
    plt.show()

    bn_afterclip_1_np = np.load('bn_afterclip_1_2.npy')
    bn_afterclip_1_tr = torch.load(
        'bn_afterclip_1_2.pt').detach().cpu().numpy()

    print('Error')
    print(((bn_afterclip_1_np[:, 0] - bn_afterclip_1_tr[:, 0]) ** 2).mean())

    plt.plot(bn_afterclip_1_np[:, 0])
    plt.plot(bn_afterclip_1_tr[:, 0])
    plt.title('bn_afterclip_1_2')
    plt.show()

    bn_afterclip_1_np = np.load('bn_afterbatch_1.npy')
    bn_afterclip_1_tr = torch.load('bn_afterbatch_1.pt').detach().cpu().numpy()

    print('Error')
    print(((bn_afterclip_1_np - bn_afterclip_1_tr[:, 0]) ** 2).mean())

    plt.plot(bn_afterclip_1_np)
    plt.plot(bn_afterclip_1_tr[:, 0])
    plt.title('bn_afterbatch_1')
    plt.show()

    bn_afterclip_1_np = np.load('bn_afterbatch_2.npy')

    print('Error')
    print(((bn_afterclip_1_np - bn_afterclip_1_tr[:, 1]) ** 2).mean())

    plt.plot(bn_afterclip_1_np)
    plt.plot(bn_afterclip_1_tr[:, 1])
    plt.title('bn_afterbatch_2')
    plt.show()

    bn_afterclip_1_np = np.load('bn_aftercv_1.npy')
    bn_afterclip_1_tr = torch.load('bn_aftercv_1.pt').detach().cpu().numpy()

    print('Error')
    print(((bn_afterclip_1_np - bn_afterclip_1_tr[:, 0]) ** 2).mean())

    plt.plot(bn_afterclip_1_np)
    plt.plot(bn_afterclip_1_tr[:, 0])
    plt.title('bn_aftercv_1')
    plt.show()

    bn_afterclip_1_np = np.load('bn_aftercv_2.npy')
    print('Error')
    print(((bn_afterclip_1_np - bn_afterclip_1_tr[:, 1]) ** 2).mean())

    plt.plot(bn_afterclip_1_np)
    plt.plot(bn_afterclip_1_tr[:, 1])
    plt.title('bn_aftercv_2')
    plt.show()

    l1_np = np.load('layer_1_output_processed.npy')
    l1_tr = torch.load('layer_1_output_processed.pt').detach().cpu().numpy()

    print('Error')
    print(((l1_np[:, 0] - l1_tr[:, 0]) ** 2).mean())

    plt.plot(l1_np[:, 14 + 3])
    plt.plot(l1_tr[:, 0])
    plt.show()

    plt.plot(l1_np[:, 14 + 4])
    plt.plot(l1_tr[:, 1])
    plt.show()

    l2_1_np = np.load('layer_2_output_2.npy')
    l2_1_tr = torch.load('layer_2_output_2.pt').detach().cpu().numpy()

    print('Error')
    print(((l2_1_np[:, 0] - l2_1_tr[:, 0]) ** 2).mean())

    plt.plot(l2_1_np[:, 0])
    plt.plot(l2_1_tr[:, 0])
    plt.show()

    l2_2_np = np.load('layer_2_output_2.npy')
    l2_2_tr = torch.load('layer_2_output_2.pt').detach().cpu().numpy()

    print('Error')
    print(((l2_2_np[:, 0] - l2_2_tr[:, 0]) ** 2).mean())

    plt.plot(l2_2_np[:, 0])
    plt.plot(l2_2_tr[:, 0])
    plt.show()

    l2_np = np.load('layer_2_output_processed.npy')
    l2_tr = torch.load('layer_2_output_processed.pt').detach().cpu().numpy()

    print('Error')
    print(((l2_np[:, 0] - l2_tr[:, 0]) ** 2).mean())

    plt.plot(l2_np[:, 28 + 3])
    plt.plot(l2_tr[:, 0])
    plt.show()

    plt.plot(l2_np[:, 28 + 4])
    plt.plot(l2_tr[:, 1])
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    import torch
    from bspytasks.utils.excel import load_bn_values
    import matplotlib.pyplot as plt

    task = RingClassificationTask(load_configs('configs/tasks/ring/template_gd_architecture.json'))
    result = task.run_task()
    task.close_test()

    # excel = pd.read_pickle(os.path.join(task.configs["results_base_dir"], 'results.pkl'))

    error = task.validate_task(excel, use_torch=False)
    print(f'Error: {error}')
