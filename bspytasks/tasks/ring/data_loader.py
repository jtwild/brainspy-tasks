import numpy as np
from bspyproc.utils.waveform import generate_waveform, generate_mask
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.input import normalise, map_to_voltage
from bspytasks.utils.datasets import sort, filter_and_reverse, subsample

import sklearn.datasets as ds
# @todo: This data should come from the model
MAX_INPUT_VOLT = np.asarray([0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3])
MIN_INPUT_VOLT = np.asarray([-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7])
VOLTAGE_LIMIT_REDUCTION = 0.1


class RingDataLoader():

    def __init__(self, configs):
        self.configs = configs

    def generate_data(self, processor_configs, ring_configs):
        inputs, targets = ds.make_circles(n_samples=ring_configs['sample_no'], factor=0.5, noise=ring_configs['noise'])

        # if inputs[targets == 0].max() > inputs[targets == 0].max():
        #     i = 1
        # else:
        #     i = 0
        # inputs[targets == i] = (inputs[targets == i] * (ring_configs['gap'] + 0.5))

        inputs[targets == 0], inputs[targets == 1] = subsample(inputs[targets == 0], inputs[targets == 1])
        inputs[targets == 0], inputs[targets == 1] = sort(inputs[targets == 0], inputs[targets == 1])
        inputs, targets = filter_and_reverse(inputs[targets == 0], inputs[targets == 1])
        inputs, targets, mask = self.process_data(inputs, targets, processor_configs)
        return inputs, targets, mask

    def get_data(self, processor_configs):
        with np.load(self.configs['ring_data_path']) as data:
            inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            print('Input shape: ', inputs.shape)
            targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)

        return self.process_data(inputs, targets, processor_configs=processor_configs)

    def process_data(self, inputs, targets, processor_configs):
        inputs = self.process_inputs(inputs, processor_configs)
        mask = generate_mask(targets, processor_configs['waveform']['amplitude_lengths'], slope_lengths=processor_configs['waveform']['slope_lengths'])
        targets = self.process_targets(targets, processor_configs)

        return inputs, targets, mask

    def process_inputs(self, inputs, processor_configs):
        assert inputs.shape[1] == len(processor_configs['input_indices'])
        for i in range(inputs.shape[1]):
            inputs[:, i] = normalise(inputs[:, i])
            inputs[:, i] = map_to_voltage(inputs[:, i],
                                          MIN_INPUT_VOLT[processor_configs['input_indices'][i]],
                                          MAX_INPUT_VOLT[processor_configs['input_indices'][i]])
        inputs = self.generate_data_waveform(inputs, processor_configs['waveform']['amplitude_lengths'], processor_configs['waveform']['slope_lengths'])
        if processor_configs["platform"] == 'simulation' and processor_configs["network_type"] == 'dnpu':
            return TorchUtils.get_tensor_from_numpy(inputs)
        return inputs

    def process_targets(self, targets, processor_configs):
        mask = (targets == 1)
        targets[targets == 0] = 1
        targets[mask] = 0
        targets = np.asarray(generate_waveform(targets, processor_configs['waveform']['amplitude_lengths'], processor_configs['waveform']['slope_lengths'])).T
        if processor_configs["platform"] == 'simulation' and processor_configs["network_type"] == 'dnpu':
            return TorchUtils.get_tensor_from_numpy(targets)
        return targets

    def generate_data_waveform(self, data, amplitude_lengths, slope_lengths):
        data_waveform = np.array([])
        for i in range(data.shape[1]):
            d_waveform = generate_waveform(data[:, i], amplitude_lengths, slope_lengths=slope_lengths)
            if data_waveform.shape == (0,):
                data_waveform = np.concatenate((data_waveform, d_waveform))
            else:
                data_waveform = np.vstack((data_waveform, d_waveform))

        return data_waveform.T
