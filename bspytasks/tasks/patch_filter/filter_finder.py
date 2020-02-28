# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:38:28 2020

@author: Jochem

    #TODO: use the new is_main idea to save data, also. In general, save data
    #TODO: fix loss function such that there is not such a large bias for large output currents? Or punish hjigh outputs currents?
"""
from bspyalgo.algorithm_manager import get_algorithm  # to get either the GA or the GD algo
from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager  # To generate the binary inputs for a patch
import matplotlib.pyplot as plt  # for plotting final figures, and saving them
import numpy as np
import os  # for saving data.
from bspyalgo.utils.io import create_directory, create_directory_timestamp, save  # For saving data
from bspytasks.utils.excel import ExcelFile # For storing data to be saved
import time  # for timestamps
#from bspyproc.bspyproc import get_processor # Only required for validation, elsewise this is called via get_algorithm
#from bspyalgo.utils.io import create_directory  # to create directories for saving
#from bspyproc.utils.pytorch import TorchUtils

class FilterFinder():
# The surrogate model is defined via config template
# The loss is defined via the algorithm, for example via gd.py or ga.py
# The optimizer is defined via the algorithm (via configs), for example via gd.py or ga.py
# Trainable parameters are defined via the config file, as the control electrodes.
    def __init__(self, configs, is_main=True):
        # Create directory structure if this is the main function.
        self.configs = configs
        self.excel_file = None    # Not sure why it is done like this, but I copied the structure defined by Unai in capacity test
        self.is_main = is_main
        self.input_dim = len( configs['algorithm_configs']['processor']['input_indices'] )
        self.init_dirs(self.input_dim)  # Initialize directory and excel files

        # And load other relevant config parameters
        self.load_methods(self.configs)
        if 'validation' in self.configs:
            raise Warning('Validation not implemented. Ignoring!')
        self.max_attempts = self.configs['max_attempts']
        self.show_plots = self.configs['show_plots']
        self.algorithm = get_algorithm(configs['algorithm_configs'])  # An instance of GD or GA, loading all algorithm related parameters.
        self.algorithm.init_dirs(self.base_dir)
        self.save_plot = self.configs['save_plot']
        # Check if correct loss/fitness function is defined
        #if self.configs['algorithm_configs']['algorithm'] == 'gradient_descent':
        #    key = 'loss_function'
        #elif self.configs['algorithm_configs']['algorithm'] == 'genetic':
        #    key = 'fitness_function_type'
        #else:
        #    raise ValueError('Selected algorithm not tested/implemented.')
        #if configs['algorithm_configs']['hyperparameters'][key] != 'sigmoid_distance':
        #    raise ValueError('For now, only implemented with sigmoidal distance loss/fitness function.')
        #else:
 #               self.algorithm = get_algorithm(configs['algorithm_configs'])  # An instance of GD or GA, loading all algorithm related parameters.

    def init_dirs(self, input_dim):
        # Main excel file, in the primary directory defined by configs
        self.init_main_file()

        # specific excel file for this run, in a timestamped directory.
        results_folder_name = f'patch_filter_{input_dim}_points'
        file_name = "patch_filter_results.xlsx"
        base_dir = self.configs['results_base_dir']
        if self.is_main:
            base_dir = create_directory_timestamp(base_dir, results_folder_name)
            self.excel_file = ExcelFile(os.path.join(base_dir, file_name))
        else:
            if self.excel_file is None:
                self.excel_file = ExcelFile(os.path.join(base_dir, file_name))
            base_dir = os.path.join(base_dir, results_folder_name)
        self.init_excel_file()
        self.base_dir = base_dir

        self.configs['results_base_dir'] = self.base_dir
        self.configs['algorithm_configs']['results_base_dir'] = self.base_dir
        self.configs_dir = os.path.join(self.base_dir, 'configs.json')
        return base_dir

    def init_excel_file(self):
        #column_names = ['number of input points','minimum seperation','average seperation','minimum current', 'maximum current','input electrodes','control electrodes','control voltages','input voltages','output currents']
        self.excel_file.init_data([''])
        self.excel_file.reset()

    def init_main_file(self):
        # Initializes a main file which contains most important results of multiple runs.
        # so data gets appended to this file instead of overwritten
        self.starttime = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
        self.main_dir = self.configs['main_results_dir']
        create_directory(self.main_dir)
        filename = 'filter_finder_collective_results.xlsx'
        self.main_file = ExcelFile(os.path.join(self.main_dir, filename), overwrite=False)
        self.main_file_keys = ['timestamp', '# input dimensions', 'algorithm', 'loss function',
                               'min seperation', 'min current', 'max current', 'attempts','epochs',
                               'batch size', 'learning rate','loss value','output points', 'input points', 'results directory']
        self.main_file.init_data(self.main_file_keys)
        self.main_file.reset()

    def load_methods(self, configs):
        if configs['algorithm_configs']['processor']['platform'] == 'simulation':
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

    def plot_best_filter(self, excel_results, index, show_plots=True, save_plot=True):
        # Setup axes
        fig, axs = plt.subplots(1,2, sharey=False)
        y = excel_results[index]['best_output']
        inputs = excel_results[index]['inputs'].tolist()

        # Plot lines in output current
        for i in range( len(y) ):
            axs[0].plot([0,1], [y[i], y[i]] )
            text = ''.join(str(round(e,2))+', ' for e in inputs[i] )
            axs[0].annotate(text, [0,y[i]])
        # Pplot of the loss over time
        axs[1].plot(excel_results[index]['performance_history'])

        #Formatting:
        axs[1].grid(which='both')
        axs[0].set_ylabel('Current (nA)')
        axs[0].grid(which='major')
        axs[0].set_xticks([])
        fig.suptitle(f"Best output for input dimension {len(inputs[0])} \n"
                     f"Minimum nearest neighbour distance: {np.round( excel_results[index]['dist']['min'], 4)} \n"
                     f"Control voltages: {str(np.round(excel_results[index]['control_voltages'][0],3))} V")

        # Show if required
        if show_plots:
            plt.show()
        # Save plot if required:
        if save_plot:
            plt.savefig( os.path.join(self.base_dir, 'best_output.pdf') )

    def fill_main_file(self):
        # Keys: ['time', '# input dimensions', 'algorithm', 'loss function', 'min seperation', 'min current',
        # 'max current', 'attempts','epochs','batch size', 'learning rate','loss value',
        # 'input points', 'output points']
        main_dict = dict()
        main_dict['timestamp'] = self.starttime
        main_dict["# input dimensions"] = self.input_dim
        # algo done at done at end
        # loss function done at end
        main_dict['min seperation'] = self.excel_results[self.best_attempt_index]['dist']['min']
        main_dict['min current'] = np.min(self.excel_results[self.best_attempt_index]['best_output'])
        main_dict['max current'] = np.max(self.excel_results[self.best_attempt_index]['best_output'])
        main_dict['attempts'] = self.max_attempts
        main_dict['epochs'] = self.configs['algorithm_configs']['hyperparameters']['nr_epochs']
        main_dict['batch size'] = self.configs['algorithm_configs']['hyperparameters']['batch_size']
        main_dict['learning rate'] = self.configs['algorithm_configs']['hyperparameters']['learning_rate']
        # loss value done at end
        main_dict['input points'] = self.excel_results[self.best_attempt_index]['inputs'].tolist()
        main_dict['output points'] = self.excel_results[self.best_attempt_index]['best_output'].tolist()
        main_dict['results directory'] = self.base_dir
        # Below fillings depend on the specific algoirhtm.
        algo = self.configs['algorithm_configs']['algorithm']
        main_dict['algorithm'] = algo
        if algo == 'gradient_descent':
            loss_key = 'loss_function'
            loss_fn_filterer = np.nanmin  # Because we need smallest loss
        elif algo == 'genetic':
            Loss_key = 'fitness_function_type'
            loss_fn_filterer = np.nanmax  # BEcause we need highest fitness
        main_dict['loss function'] = self.configs['algorithm_configs']['hyperparameters'][loss_key]
        main_dict['loss value'] = loss_fn_filterer(self.excel_results[self.best_attempt_index]['performance_history'])
        # Finally, add one row to the excel file data:
        self.main_file.add_result(main_dict)
        return main_dict



# %% find_filter is the main method. All else above is required for this function
    def find_filter(self):
        # Save configs for reproducability
        if self.is_main:
            save(mode='configs', file_path=self.configs_dir, data=self.configs)

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
        self.excel_results = []
        for attempt in range(self.max_attempts):
            print(f'\nAttempt {attempt+1} of {self.max_attempts}.')
            self.excel_results.append( self.find_filter_core(inputs) )
            distances = self.excel_results[attempt]['best_output'] - self.excel_results[attempt]['best_output'].T
            np.fill_diagonal(distances, np.nan)  # ignore diagonal, distance to itself always zero
            distance_nearest = np.nanmin( abs(distances), axis=0 )
            self.excel_results[attempt]['dist'] = dict()  # intialize new key
            self.excel_results[attempt]['dist']['nn'] = distance_nearest  # all nearest neighbour distances
            self.excel_results[attempt]['dist']['avg'] = np.mean( distance_nearest )  # average nearest neighbour distance
            self.excel_results[attempt]['dist']['min'] = np.nanmin( distance_nearest )  # minimal nearest neighbour distanc
            self.excel_file.add_result(self.excel_results[attempt])

        # Find best attempt:
        performance = []
        for results in self.excel_results:
            performance.append( np.nanmin( results['performance_history'] ) )
        self.best_attempt_index = np.argmin(performance)

        # Save data to specific excel file
        aux = self.excel_file.data.copy()
        tab_name = 'Patch_Filter'
        self.excel_file.save_tab(tab_name, data=aux)
        self.excel_file.close_file()

        # Save data to main excel file
        self.fill_main_file()
        main_data = self.main_file.data.copy()
        tab_name = 'main'
        self.main_file.save_tab(tab_name, data=main_data)
        self.main_file.close_file()

        # Plot best output
        self.plot_best_filter(self.excel_results, self.best_attempt_index, self.show_plots, self.save_plot)
        return self.excel_results

#%% Testing
if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('configs/tasks/filter_finder/template_ff_gd.json')
    task = FilterFinder(configs['filter_finder'], is_main=True) #initialize class
    excel_results = task.find_filter()
