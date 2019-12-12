import numpy as np
import bspyproc.utils.waveform as waveform
from bspyproc.utils.pytorch import TorchUtils


ZERO = -0.7
ONE = 0.7


class VCDimDataManager():

    def __init__(self, configs):
        self.amplitude_lengths = configs['boolean_gate_test']['algorithm_configs']['processor']['waveform']['amplitude_lengths']
        self.slope_lengths = configs['boolean_gate_test']['algorithm_configs']['processor']['waveform']['slope_lengths']
        self.use_waveform = True
        if configs['boolean_gate_test']['algorithm_configs']['algorithm'] == 'gradient_descent' and configs['boolean_gate_test']['algorithm_configs']['processor']['platform'] == 'simulation':
            self.use_torch = True
        else:
            self.use_torch = False

    def get_data(self, vc_dimension, verbose=True):
        readable_inputs, transformed_inputs = self.get_inputs(vc_dimension)
        readable_targets, transformed_targets = self.get_targets(vc_dimension, verbose)
        mask = waveform.generate_mask(readable_targets[1], self.amplitude_lengths, slope_lengths=self.slope_lengths)  # Chosen readable_targets[1] because it might be better for debuggin purposes. Any other label or input could be taken.
        readable_targets, transformed_targets, found = self.get_dictionaries(readable_inputs, transformed_inputs, readable_targets, transformed_targets)
        return readable_inputs, transformed_inputs, readable_targets, transformed_targets, found, mask

    def get_inputs(self, vc_dimension):
        readable_inputs = self.generate_test_inputs(vc_dimension)
        if self.use_waveform:
            transformed_inputs = self.generate_inputs_waveform(readable_inputs)
        else:
            transformed_inputs = readable_inputs
        if self.use_torch:
            transformed_inputs = TorchUtils.get_tensor_from_numpy(transformed_inputs)

        return readable_inputs, transformed_inputs

    def get_dictionaries(self, readable_inputs, transformed_inputs, readable_targets, transformed_targets):
        readable_targets_dict = {}
        transformed_targets_dict = {}
        found_dict = {}

        for i in range(len(readable_targets)):
            key = str(readable_targets[i])
            readable_targets_dict[key] = readable_targets[i]
            transformed_targets_dict[key] = transformed_targets[i]  # , :]  # transformed_targets[i]
            found_dict[key] = False
            # readable_inputs_dict[key] = readable_inputs[i]
            # transformed_inputs_dict[key] = transformed_inputs[i]

        return readable_targets_dict, transformed_targets_dict, found_dict

    def get_targets(self, vc_dimension, verbose=True):
        readable_targets = self.generate_test_targets(vc_dimension, verbose)
        if self.use_waveform:
            transformed_targets = self.generate_targets_waveform(readable_targets)
        else:
            transformed_targets = readable_targets
        if self.use_torch:
            transformed_targets = TorchUtils.get_tensor_from_numpy(transformed_targets)

        return readable_targets, transformed_targets

    def generate_test_inputs(self, vc_dimension):
        # @todo create a function that automatically generates non-linear inputs
        try:
            if vc_dimension == 4:
                return [[ZERO, ZERO, ONE, ONE], [ONE, ZERO, ONE, ZERO]]
            elif vc_dimension == 5:
                return [[ZERO, ZERO, ONE, ONE, -0.35],
                        [ONE, ZERO, ONE, ZERO, 0.0]]
            elif vc_dimension == 6:
                return [[ZERO, ZERO, ONE, ONE, -0.35, 0.35],
                        [ONE, ZERO, ONE, ZERO, 0.0, 0.0]]
            elif vc_dimension == 7:
                return [[ZERO, ZERO, ONE, ONE, -0.35, 0.35, 0.0],
                        [ONE, ZERO, ONE, ZERO, 0.0, 0.0, 1.0]]
            elif vc_dimension == 8:
                return [[ZERO, ZERO, ONE, ONE, -0.35, 0.35, 0.0, 0.0],
                        [ONE, ZERO, ONE, ZERO, 0.0, 0.0, 1.0, -1.0]]
            else:
                raise VCDimensionException()
        except VCDimensionException:
            print(
                'Dimension Exception occurred. The selected VC Dimension is %d Please insert a value between ' % vc_dimension)

    def generate_inputs_waveform(self, inputs):
        inputs_waveform = np.array([])
        for inp in inputs:
            input_waveform = waveform.generate_waveform(inp, self.amplitude_lengths, slope_lengths=self.slope_lengths)
            if inputs_waveform.shape == (0,):
                inputs_waveform = np.concatenate((inputs_waveform, input_waveform))
            else:
                inputs_waveform = np.vstack((inputs_waveform, input_waveform))
        # if len(inputs_waveform.shape) == 1:
        #    inputs_waveform = inputs_waveform[:, np.newaxis]
        return inputs_waveform.T  # device_model --> (samples,dimension) ; device --> (dimensions,samples)

    def generate_test_targets(self, vc_dimension, verbose=True):
        # length of list, i.e. number of binary targets
        binary_target_no = 2**vc_dimension
        assignments = []
        list_buf = []

        # construct assignments per element i
        if verbose:
            print('===' * vc_dimension)
            print('ALL BINARY LABELS:')
        level = int((binary_target_no / 2))
        while level >= 1:
            list_buf = []
            buf0 = [0] * level
            buf1 = [1] * level
            while len(list_buf) < binary_target_no:
                list_buf += (buf0 + buf1)
            assignments.append(list_buf)
            level = int(level / 2)

        binary_targets = np.array(assignments).T
        if verbose:
            print(binary_targets)
            print('===' * vc_dimension)
        return binary_targets

    def generate_targets_waveform(self, targets):
        targets_waveform = np.array([])[:, np.newaxis]
        for target in targets:
            target_waveform = waveform.generate_waveform(target, self.amplitude_lengths, slope_lengths=self.slope_lengths)
            waveform_length = len(target_waveform)
            target_waveform = target_waveform[:, np.newaxis]
            # targets_waveform.append(targets_waveform)
            if targets_waveform.shape == (0, 1):
                targets_waveform = np.concatenate((targets_waveform, target_waveform))
            else:
                targets_waveform = np.vstack((targets_waveform, target_waveform))
        return targets_waveform.reshape(len(targets), waveform_length, 1)


class VCDimensionException(Exception):
    """Exception: It does not exist an implementation of such VC Dimension."""
    pass
