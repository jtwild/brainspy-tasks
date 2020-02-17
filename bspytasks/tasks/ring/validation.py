from bspyproc.bspyproc import get_processor
from bspytasks.tasks.ring.classifier import RingClassificationTask
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspyproc.utils.waveform import generate_waveform, generate_mask

import matplotlib.pyplot as plt
import numpy as np


class RingClassifierValidator():

    def __init__(self, configs, processor=None):
        self.configs = configs
        self.data_loader = RingDataLoader(configs)
        self.validation_processor = get_processor(configs['validation']['processor'])
        if processor is None:
            self.processor = get_processor(configs['algorithm_configs']['processor'])
        else:
            self.processor = processor

    def get_model_output(self):
        self.processor.load_state_dict(model.copy())
        self.processor.eval()
        model_output = self.processor.forward(results['inputs']).detach().cpu().numpy()
        return generate_waveform(model_output[:, 0], self.configs['validation']['processor']['waveform']
                                 ['amplitude_lengths'], self.configs['validation']['processor']['waveform']['slope_lengths'])

    def get_real_output(self):
        self.validation_processor.load_state_dict(model.copy())
        inputs, mask = self.get_validation_inputs(results)
        return self.validation_processor.get_output_(inputs, mask)[:, 0], mask

    def validate(self, results, model):
        model_output = self.get_model_output()
        real_output, mask = self.get_real_output()
        self.plot_validation_results(model_output[mask], real_output[mask])

    def get_validation_inputs(self, results):
        inputs = results['inputs'].cpu().numpy()
        processor_configs = self.configs['validation']['processor']
        inputs_1 = generate_waveform(inputs[:, 0], processor_configs['waveform']['amplitude_lengths'], slope_lengths=processor_configs['waveform']['slope_lengths'])
        inputs_2 = generate_waveform(inputs[:, 1], processor_configs['waveform']['amplitude_lengths'], slope_lengths=processor_configs['waveform']['slope_lengths'])
        inputs = np.asarray([inputs_1, inputs_2]).T
        mask = generate_mask(results['targets'].cpu().numpy(), processor_configs['waveform']['amplitude_lengths'], slope_lengths=processor_configs['waveform']['slope_lengths'])
        return inputs, mask

    def plot_validation_results(self, model_output, real_output):
        error = ((model_output - real_output) ** 2).mean()
        print(f'Total Error: {error}')

        fig = plt.figure()
        plt.title('Comparison between Processor and DNPU')
        fig.suptitle(f'MSE: {error}', fontsize=10)
        plt.plot(model_output)
        plt.plot(real_output, '-.')
        plt.ylabel('Current (nA)')
        plt.xlabel('Time')

        plt.legend(['Simulation', 'Validation'])
        # if save_dir is not None:
        #     plt.savefig(save_dir)
        # if show_plot:
        plt.show()
        plt.close()


if __name__ == '__main__':
    import torch
    import pickle
    from bspyalgo.utils.io import load_configs
    model = torch.load('model.pth')
    results = pickle.load(open('best_output_results.pkl', "rb"))
    configs = load_configs('ring_classification_configs.json')

    val = RingClassifierValidator(configs)
    val.validate(results, model)
