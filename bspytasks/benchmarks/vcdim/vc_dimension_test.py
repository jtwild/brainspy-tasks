#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:18:29 2018
Wrapper to measure the VC dimension of a device using the measurement script measure_VCdim.py
This wrapper creates the binary gates for N points and for each gate it finds the control voltages.
If successful (measured by a threshold on the correlation and by the perceptron accuracy), the entry 1 is set in a vector corresponding to all gatelings.
@author: hruiz and ualegre
Adjusted bu Jochem for multi dimensional input
"""

import os
import numpy as np
import time
from matplotlib import pyplot as plt
from bspyalgo.utils.io import create_directory, create_directory_timestamp
from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager
from bspytasks.tasks.boolean.gate_finder import BooleanGateTask
from bspytasks.utils.excel import ExcelFile


class VCDimensionTest():

    def __init__(self, configs, is_main=True):
        self.data_manager = VCDimDataManager(configs)
        self.base_dir = configs['results_base_dir']
        self.threshold_parameter = configs['threshold_parameter']
        self.show_plots = configs['show_plots']
        self.is_main = is_main
        self.load_boolean_gate_configs(configs['boolean_gate_test'])
        self.configs = configs

    def load_boolean_gate_configs(self, configs):
        self.boolean_gate_test_configs = configs
        self.load_algorithm_configs(configs)
        self.excel_file = None

    def load_algorithm_configs(self, configs):
        self.amplitude_lengths = configs['algorithm_configs']['processor']['waveform']['amplitude_lengths']
        self.slope_lengths = configs['algorithm_configs']['processor']['waveform']['slope_lengths']
        self.input_dim = len(configs['algorithm_configs']['processor']['input_indices'])
        self.control_dim = configs['algorithm_configs']['processor']["input_electrode_no"] - self.input_dim

    def init_dirs(self, vc_dimension):
        results_folder_name = f'vc_dimension_{vc_dimension}'

        if self.is_main:
            base_dir = create_directory_timestamp(self.base_dir, results_folder_name)
            self.excel_file = ExcelFile(os.path.join(base_dir, 'capacity_test_results.xlsx'))
        else:
            if self.excel_file is None:
                self.excel_file = ExcelFile(os.path.join(self.base_dir, 'capacity_test_results.xlsx'))
            base_dir = os.path.join(self.base_dir, results_folder_name)  # 'dimension_' + str(vc_dimension))
        self.boolean_gate_test_configs['results_base_dir'] = base_dir
        return base_dir

    def init_test(self, vc_dimension, validation=False):
        self.vc_dimension = vc_dimension
        self.threshold = self.calculate_threshold()
        self.readable_inputs, self.transformed_inputs, readable_targets, transformed_targets, found, self.mask = self.data_manager.get_data(vc_dimension, validation=validation)
        base_dir = self.init_dirs(vc_dimension)
        self.boolean_gate_test_configs['algorithm_configs']['processor']['shape'] = self.data_manager.get_shape(vc_dimension, validation=False)
        self.boolean_gate_test_configs['validation']['processor']['shape'] = self.data_manager.get_shape(vc_dimension, validation=True)
        self.boolean_gate_task = BooleanGateTask(self.boolean_gate_test_configs, is_main=False)
        self.init_excel_file(readable_targets, transformed_targets, found)
        self.init_custom_file(base_dir)
        return base_dir

    def init_excel_file(self, readable_targets, transformed_targets, found):
        column_names = ['gate', 'found', 'accuracy', 'best_output', 'control_voltages', 'correlation', 'best_performance', 'validation_error', 'encoded_gate']
        self.excel_file.init_data(column_names, readable_targets)
        self.excel_file.reset()
        self.excel_file.insert_column('gate', readable_targets)
        self.excel_file.insert_column('encoded_gate', transformed_targets)
        self.excel_file.insert_column('found', found)

    def calculate_threshold(self):
        return (1 - (self.threshold_parameter / self.vc_dimension)) * 100.0

    def run_test(self, vc_dimension, validate=False):
        base_dir = self.init_test(vc_dimension)
        print('---------------------------------------------')
        print(f'    VC DIMENSION {str(vc_dimension)} TEST')
        print('---------------------------------------------')

        number_gates = 2**vc_dimension
        gate_array = np.zeros((number_gates, vc_dimension))
        accuracy_array = np.zeros(number_gates)
        performance_array = np.zeros_like(accuracy_array)
        found_array = np.zeros_like(accuracy_array)
        correlation_array = np.zeros_like(accuracy_array)
        control_voltages_per_gate = np.zeros((number_gates, self.control_dim))  # TODO: un-hard code nr. dimensions

        length_waveform = len(self.transformed_inputs)
        output_array = np.zeros((number_gates, length_waveform))
        targets_array = np.zeros_like(output_array)
        for nr, r in enumerate(self.excel_file.data.iterrows()):
            _, row = r
            excel_results = self.boolean_gate_task.find_gate(self.transformed_inputs, row['gate'], row['encoded_gate'], self.mask, self.threshold)
            self.excel_file.add_result(excel_results, row['gate'])
            # Update custom excelm file
            self.update_custom_file(row['gate'])
            # Collect results
            gate_array[nr] = excel_results['gate']
            accuracy_array[nr] = excel_results['accuracy']
            performance_array[nr] = excel_results['best_performance']
            found_array[nr] = excel_results['found']
            correlation_array[nr] = excel_results['correlation']
            control_voltages_per_gate[nr] = excel_results['control_voltages']
            if type(row['encoded_gate']) is np.ndarray:
                targets_array[nr] = row['encoded_gate'][:, 0]
            else:
                targets_array[nr] = row['encoded_gate'].cpu().numpy()[:, 0]
            if type(excel_results['control_voltages']) is float:
                output_array[nr] = excel_results['best_output']
            else:
                output_array[nr] = excel_results['best_output'][:, 0]
        mask = self.mask
        if type(row['encoded_gate']) is np.ndarray:
            inputs = self.transformed_inputs
        else:
            inputs = self.transformed_inputs.detach().cpu().numpy()
        # if validate:
        #     for _, row in self.excel_file.data.iterrows():
        #         if row['control_voltages'] is not np.nan:
        #             validation_error = self.boolean_gate_task.validate_gate(row['gate'], self.transformed_inputs, row['control_voltages'], row['best_output'], self.mask, base_dir)
        #             row['validation_error'] = validation_error
        #             self.excel_file.add_result(row.to_dict(), row['gate'])

        capacity = np.mean(found_array)
        result = self.close_test(base_dir)
        self.save_custom_file()
        os.mkdir(os.path.join(base_dir, 'validation'))
        numpy_file = os.path.join(base_dir, 'validation', 'result_arrays')
        np.savez(numpy_file,
                 gate_array=gate_array,
                 accuracy_array=accuracy_array,
                 performance_array=performance_array,
                 found_array=found_array,
                 correlation_array=correlation_array,
                 control_voltages_per_gate=control_voltages_per_gate,
                 targets_array=targets_array,
                 output_array=output_array, inputs=inputs, mask=mask)
        print('---------------------------------------------')
        if result:
            print(f'VC DIMENSION {str(vc_dimension)} TEST VEREDICT: PASSED')
        else:
            print(f'VC DIMENSION {str(vc_dimension)} TEST VEREDICT: FAILED')
        print('---------------------------------------------')

        return capacity, accuracy_array, performance_array, correlation_array

    def get_not_found_gates(self):
        return self.excel_file.data['gate'].loc[self.excel_file.data['found'] == False].size  # noqa: E712

    def close_test(self, base_dir):
        aux = self.excel_file.data.copy()
        aux.index = range(len(aux.index))
        tab_name = 'VC Dimension ' + str(self.vc_dimension) + ' Threshold ' + str(round(self.threshold, 4))
        # rounding required for 1/3=0.3333.... type numbers with too much decimals to place in an excel workbook tab name
        self.excel_file.save_tab(tab_name, data=aux)
        self.plot_results(base_dir)
        self.close_algorithm()
        return self.oracle()

    def close_algorithm(self):
        try:
            self.algorithm.close()
        except AttributeError:
            print('\nThere is no closing function for the current algorithm configuration. Skipping. \n')

    def plot_results(self, base_dir, plot_name='_plot', extension='png'):
        plt.figure()
        fitness_classifier = self.excel_file.data['best_performance'].to_numpy()
        plt.plot(fitness_classifier, self.excel_file.data['accuracy'].to_numpy(), 'o')
        plt.plot(np.linspace(np.nanmin(fitness_classifier),
                             np.nanmax(fitness_classifier)),
                 self.threshold * np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness / Performance')
        plt.ylabel('Accuracy')

        # create_directory(path)
        plt.savefig(os.path.join(base_dir, 'dimension_' + str(self.vc_dimension) + plot_name + '.' + extension))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def oracle(self):
        return self.excel_file.data.loc[self.excel_file.data['found'] == False].size == 0  # noqa: E712

    # def plot_output(self, row):
    #     path = os.path.join(self.output_dir + 'dimension_' + str(self.vc_dimension), self.boolean_gate_task.is_found(row['found']))
    #     create_directory(path)
    #     # self.boolean_gate_task.plot_gate(row, self.mask, self.show_plots, os.path.join(path, str(row['gate']) + '_' + self.test_data_plot_name))

    def close_results_file(self):
        self.excel_file.close_file()

    def init_custom_file(self, base_dir):
        column_names = ['timestamp', 'gate', 'found', 'accuracy', 'final_output', 'control_voltages',
                        'correlation', 'final_performance',
                        'input_electrodes', 'input_voltages', 'num_levels', 'voltage_intervals',
                        'min_gap', 'min_output', 'max_output',
                        'loss_function', 'learning_rate', 'max_attempts', 'nr_epochs']
        self.custom_file = ExcelFile(os.path.join(base_dir, 'custom_capacity_test_results.xlsx'))
        self.custom_file.init_data(column_names)
        self.custom_file.reset()

    def update_custom_file(self, gate):
        # This function fills an excel file with any custom entries the user might require for their work.
        temp_dict = dict()
        # Update all dict keys (column names) if we did not ignore the gate
        if len(np.unique(gate)) != 1:
            temp_dict['timestamp'] = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
            temp_dict['gate'] = self.boolean_gate_task.algorithm_data.results['gate']
            temp_dict['found'] = self.boolean_gate_task.algorithm_data.results['found']
            temp_dict['accuracy'] = self.boolean_gate_task.algorithm_data.results['accuracy']
            temp_dict['final_output'] = self.boolean_gate_task.algorithm_data.results['best_output'].copy().tolist()
            temp_dict['control_voltages'] = self.boolean_gate_task.algorithm_data.results['control_voltages'].copy().tolist()
            temp_dict['correlation'] = self.boolean_gate_task.algorithm_data.results['correlation']
            temp_dict['final_performance'] = self.boolean_gate_task.algorithm_data.results['best_performance'].item()
            temp_dict['input_electrodes'] = self.configs['boolean_gate_test']['algorithm_configs']['processor']['input_indices']
            temp_dict['input_voltages'] = self.boolean_gate_task.algorithm_data.results['inputs'].cpu().detach().numpy().copy().tolist()
            temp_dict['num_levels'] = self.configs['num_levels']
            temp_dict['voltage_intervals'] = self.configs['voltage_intervals']
            temp_dict['targets'] = self.boolean_gate_task.algorithm_data.results['targets'].cpu().detach().numpy().copy().tolist()
            # The following is a beastly expression, but it is simply
            # gap = min(Class1) - max(Class0)
            # Only makes sense when a solution is found. If solution is inverted, number is negative
            temp_dict['min_gap'] = (self.boolean_gate_task.algorithm_data.results['best_output'][self.boolean_gate_task.algorithm_data.results['targets'].cpu().detach().numpy().astype(bool)].max()
                                    - self.boolean_gate_task.algorithm_data.results['best_output'][np.invert(self.boolean_gate_task.algorithm_data.results['targets'].cpu().detach().numpy().astype(bool))].min())
            temp_dict['min_output'] = min(temp_dict['final_output'])[0]  # take the number instead of the list
            temp_dict['max_output'] = max(temp_dict['final_output'])[0]
            temp_dict['loss_function'] = self.configs['boolean_gate_test']['algorithm_configs']['hyperparameters']['loss_function']
            temp_dict['learning_rate'] = self.configs['boolean_gate_test']['algorithm_configs']['hyperparameters']['learning_rate']
            temp_dict['max_attempts'] = self.configs['boolean_gate_test']['max_attempts']
            temp_dict['nr_epochs'] = self.configs['boolean_gate_test']['algorithm_configs']['hyperparameters']['nr_epochs']
        # Save to file
        self.custom_file.add_result(temp_dict)
        return temp_dict

    def save_custom_file(self):
        aux = self.custom_file.data.copy()
        aux.index = range(len(aux.index))
        tab_name = 'VC Dimension ' + str(self.vc_dimension) + ' Threshold ' + str(round(self.threshold, 4))
        # rounding required for 1/3=0.3333.... type numbers with too much decimals to place in an excel workbook tab name
        self.custom_file.save_tab(tab_name, data=aux)
#        self.custom_file.close_file()


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('configs/benchmark_tests/capacity/template_gd.json')
    configs = configs['capacity_test']['vc_dimension_test']
    dimension = 4
    data_manager = VCDimensionTest(configs)
    data_manager.run_test(dimension, validate=True)
    data_manager.close_results_file()
