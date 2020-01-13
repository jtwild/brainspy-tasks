import numpy as np
from bspyproc.utils.waveform import generate_waveform, generate_mask
from bspyproc.utils.pytorch import TorchUtils


class RingDataLoader():

    def __init__(self, configs):
        self.configs = configs

    def get_ring_data_from_npz(self, processor_configs):
        with np.load(self.configs['ring_data_path']) as data:
            inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            print('Input shape: ', inputs.shape)
            targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)
        return self.process_npz_targets(inputs, targets, processor_configs=processor_configs)

    def get_ring_data_from_npz_2(self, processor_configs):
        with np.load(self.configs['ring_data_path']) as data:
            inputs = data['inp_wvfrm'][::self.configs['steps'], :]  # .T
            print('Input shape: ', inputs.shape)
            targets = data['target'][::self.configs['steps']]
            print('Target shape ', targets.shape)
        return self.process_npz_targets_2(inputs, targets, processor_configs=processor_configs)

    def process_npz_targets_2(self, inputs, targets, processor_configs):
        mask0 = (targets == 0)
        mask1 = (targets == 1)
        targets[mask0] = 1
        targets[mask1] = 0
        amplitude_lengths = processor_configs['waveform']['amplitude_lengths']
        slope_lengths = processor_configs['waveform']['slope_lengths']
        mask = generate_mask(targets, amplitude_lengths, slope_lengths=slope_lengths)
        # inputs = self.generate_data_waveform(inputs, amplitude_lengths, slope_lengths)
        targets = np.asarray(generate_waveform(targets, amplitude_lengths, slope_lengths)).T

        if processor_configs["platform"] == 'simulation' and processor_configs["network_type"] == 'dnpu':
            inputs = TorchUtils.get_tensor_from_numpy(inputs)
            targets = TorchUtils.get_tensor_from_numpy(targets)

        return inputs, targets, mask

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

        if processor_configs["platform"] == 'simulation' and processor_configs["network_type"] == 'dnpu':
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
