# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:38:28 2020

@author: Jochem

    #TODO: use the new is_main idea to save data, also. In general, save data
    #TODO: fix loss function such that there is not such a large bias for large output currents? Or punish hjigh outputs currents?
"""
from bspyalgo.algorithm_manager import get_algorithm  # to get either the GA or the GD algo
from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager  # To generate the binary inputs for a patch
import matplotlib.pyplot as plt
import numpy as np
import os  # for saving data.
from bspyalgo.utils.io import create_directory, create_directory_timestamp, save  # For saving data
from bspytasks.utils.excel import ExcelFile # For storing data to be saved
#from bspyproc.bspyproc import get_processor # Only required for validation, elsewise this is called via get_algorithm
#from bspyalgo.utils.io import create_directory  # to create directories for saving
#from bspyproc.utils.pytorch import TorchUtils

class FilterFinder():
# The surrogate model is loaded via ...
# The loss is defined via the algorithm, for example via gd.py or ga.py
# The optimizer is defined via the algorithm (via configs), for example via gd.py or ga.py
# Trainable parameters are defined via the config file, as the control electrodes.
    def __init__(self, configs, is_main=True):
        # Create directory structure if this is the main function.
        self.configs = configs
        self.excel_file = None    # Not sure why it is done like this, but I copied the structure defined by Unai in capacity test
        self.is_main = is_main
        self.input_dim = len( configs['filter_finder']['algorithm_configs']['processor']['input_indices'] )
        self.init_dirs(self.input_dim)  # Initialize directory and excel file.

        # And load other relevant config parameters
        self.load_methods(self.configs)
        if 'validation' in self.configs['filter_finder']:
            raise Warning('Validation not implemented. Ignoring!')
        self.max_attempts = self.configs['filter_finder']['max_attempts']
        self.show_plots = self.configs['filter_finder']['show_plots']
        self.algorithm = get_algorithm(configs['filter_finder']['algorithm_configs'])  # An instance of GD or GA, loading all algorithm related parameters.


        # Check if correct loss/fitness function is defined
        #if self.configs['filter_finder']['algorithm_configs']['algorithm'] == 'gradient_descent':
        #    key = 'loss_function'
        #elif self.configs['filter_finder']['algorithm_configs']['algorithm'] == 'genetic':
        #    key = 'fitness_function_type'
        #else:
        #    raise ValueError('Selected algorithm not tested/implemented.')
        #if configs['filter_finder']['algorithm_configs']['hyperparameters'][key] != 'sigmoid_distance':
        #    raise ValueError('For now, only implemented with sigmoidal distance loss/fitness function.')
        #else:
 #               self.algorithm = get_algorithm(configs['filter_finder']['algorithm_configs'])  # An instance of GD or GA, loading all algorithm related parameters.




    def init_dirs(self, input_dim):
        results_folder_name = f'patch_filter_{input_dim}_points'
        file_name = "patch_filter_results.xlsx"
        self.base_dir = self.configs['filter_finder']['results_base_dir']
        if self.is_main:
            base_dir = create_directory_timestamp(self.base_dir, results_folder_name)
            self.excel_file = ExcelFile(os.path.join(base_dir, file_name))
        else:
            if self.excel_file is None:
                self.excel_file = ExcelFile(os.path.join(self.base_dir, file_name))
            base_dir = os.path.join(self.base_dir, results_folder_name)
        self.configs['filter_finder']['results_base_dir'] = base_dir
        self.configs_dir = os.path.join(self.base_dir, 'test_configs.json')
        return base_dir

    def load_task_configs(self, configs):

    def load_methods(self, configs):
        if configs['filter_finder']['algorithm_configs']['processor']['platform'] == 'simulation':
            self.find_filter_core = self.optimize
        else:
            raise ValueError('Algorithm or processor not yet implemented')

    def optimize(self, inputs):
        # First do the optimization defined by this specific algorithm (GA/GD)
        algorithm_data = self.algorithm.optimize(inputs)
        algorithm_data.judge()  # Updates information in the results, such as control voltages.
        # Then generate the information regarding the results, such as performace
        # this also sets algorithm_data.results (type: excel_data, a child of a Pandas DataFrame)
        excel_results = algorithm_data.results
        return excel_results

    def plot_best_filter(self, excel_results, show_plots=True):
        performance = []
        for results in excel_results:
            performance.append( np.nanmin( results['performance_history'] ) )
        best_attempt_index = np.argmin(performance)

        fig, axs = plt.subplots(1,2, sharey=False)
        y = excel_results[best_attempt_index]['best_output']
        inputs = excel_results[best_attempt_index]['inputs'].tolist()
        for i in range( len(y) ):
            axs[0].plot([0,1], [y[i], y[i]] )
            text = ''.join(str(round(e,2))+', ' for e in inputs[i] )
            axs[0].annotate(text, [0,y[i]])
        # and a plot of the loss over time
        axs[1].plot(excel_results[best_attempt_index]['performance_history'])
        axs[1].grid(which='both')
        axs[0].set_ylabel('Current (nA)')
        axs[0].grid(which='major')
        axs[0].set_xticks([])

        fig.suptitle(f"Best output for input dimension {len(inputs[0])} \n"
                     f"Minimum nearest neighbour distance: {np.round( excel_results[best_attempt_index]['dist']['min'], 4)} \n"
                     f"Control voltages: {str(np.round(excel_results[best_attempt_index]['control_voltages'][0],3))} V")
        #TODO: add ticklabels defining the inputs
        #excel_results['inputs']int().tolist() can be used to extract the inputs in a python list.


# find_filter is the main method. All else above is required for this function
    def find_filter(self):
        # Save configs for reproducability
        if self.is_main:
            save(mode='configs', file_path=self.configs_dir, data=configs)

        # Load the required inputs:
        data_manager = VCDimDataManager(self.configs)
        #self.readable_inputs, self.transformed_inputs, readable_targets, transformed_targets, found, self.mask = data_manager.get_data(self.input_dim)
        inputs = data_manager.get_inputs(2**self.input_dim)[1]  # Get inputs from a function originally written for VC targets.
        #inputs = data_manager.get_targets(self.input_dim, verbose=False)[1]  # This calls a function orginally written to get targets for VC, but can also be used to generate boolean labels.
        #inputs = inputs.flatten(1)  # Flatten the inputs to have correct shape of [16, 2] instead of [16, 2, 1]
        #TODO: Check if readable inputs is the correct one, or if you need transformed inputs

        # Start training different initializations (attempts)
        print('--------------------------------------------------------------------')
        print(f'       Patch Filter Finder with {str(self.input_dim)} points        ')
        print('--------------------------------------------------------------------')
        excel_results = []
        for attempt in range(self.max_attempts):
            print(f'\nAttempt {attempt} of {self.max_attempts}.')
            excel_results.append( self.find_filter_core(inputs) )
            distances = excel_results[attempt]['best_output'] - excel_results[attempt]['best_output'].T
            np.fill_diagonal(distances, np.nan)  # ignore diagonal, distance to itself always zero
            distance_nearest = np.nanmin( abs(distances) )
            excel_results[attempt]['dist'] = dict()  # intialize new key
            excel_results[attempt]['dist']['nn'] = distance_nearest  # all nearest neighbour distances
            excel_results[attempt]['dist']['avg'] = np.mean( distance_nearest )  # average nearest neighbour distance
            excel_results[attempt]['dist']['min'] = np.nanmin( distance_nearest )  # minimal nearest neighbour distanc

        # Save data if required

        # Plot best output
        self.plot_best_filter(excel_results, self.show_plots)
        return excel_results

#%% Testing
if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('configs/tasks/filter_finder/template_ff_gd.json')
    task = FilterFinder(configs, is_main=True) #initialize class
    excel_results = task.find_filter()
