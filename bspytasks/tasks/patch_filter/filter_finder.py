# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:38:28 2020

@author: Jochem
"""
from bspyalgo.algorithm_manager import get_algorithm  # to get either the GA or the GD algo
from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager  # To generate the binary inputs for a patch
#from bspyproc.bspyproc import get_processor # Only required for validation, elsewise this is called via get_algorithm
#from bspyalgo.utils.io import create_directory  # to create directories for saving
#from bspyproc.utils.pytorch import TorchUtils

class FilterFinder():
# The surrogate model is loaded via ...
# The loss is defined via the algorithm, for example via gd.py or ga.py
# The optimizer is defined via the algorithm (via configs), for example via gd.py or ga.py
# Trainable parameters are defined via the config file, as the control electrodes.
    def __init__(self, configs):
        self.configs = configs
        if configs['filter_finder']['algorithm_configs']['hyperparameters']['loss_function'] != 'sigmoid_distance':
            raise ValueError('For now, only implemented with sigmoidal distance loss function.')
        else:
            self.algorithm = get_algorithm(configs['filter_finder']['algorithm_configs'])  # An instance of GD or GA, loading all algorithm related parameters.
        self.load_methods(configs)
        self.load_task_configs(configs)

    def load_task_configs(self, configs):
        if 'validation' in configs['filter_finder']:
            raise Warning('Validation not implemented. Ignoring!')
        self.input_dim = len( configs['filter_finder']['algorithm_configs']['processor']['input_indices'] )

    def load_methods(self, configs):
        if configs['filter_finder']['algorithm_configs']['algorithm'] == 'gradient_descent' and configs['filter_finder']['algorithm_configs']['processor']['platform'] == 'simulation':
            self.find_filter_core = self.optimize
        else:
            raise ValueError('Algorithm or processor not yet implemented')

    def optimize(self, inputs):
        # First do the optimization defined by this specific algorithm (GA/GD)
        algorithm_data = self.algorithm.optimize(inputs)
        # Then generate the information regarding the results, such as performace
        # this also sets algorithm_data.results (type: excel_data, a child of a Pandas DataFrame)
        excel_results = algorithm_data.results
        return excel_results

# find_filter is the main method. All else above is required for this function
    def find_filter(self):
        # Load the required inputs:
        data_manager = VCDimDataManager(self.configs)
        #self.readable_inputs, self.transformed_inputs, readable_targets, transformed_targets, found, self.mask = data_manager.get_data(self.input_dim)
        inputs = data_manager.get_targets(self.input_dim, verbose=False)[1]  # This calls a function orginally written to get targets for VC, but can also be used to generate boolean labels.
        #TODO: Check if readable inputs is the correct one, or if you need transformed inputs
        print(inputs)
        excel_results = self.find_filter_core(inputs)

#%% Testing
if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('configs/tasks/filter_finder/template_ff.json')
    task = FilterFinder(configs) #initialize class
    task.find_filter()
    #TODO: Adjust gd.py to have the targets as a possible keyword argument.
