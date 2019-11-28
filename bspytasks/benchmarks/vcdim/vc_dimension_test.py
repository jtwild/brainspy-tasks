#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:18:29 2018
Wrapper to measure the VC dimension of a device using the measurement script measure_VCdim.py
This wrapper creates the binary labels for N points and for each label it finds the control voltages.
If successful (measured by a threshold on the correlation and by the perceptron accuracy), the entry 1 is set in a vector corresponding to all labellings.
@author: hruiz and ualegre
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from bspyalgo.utils.io import create_directory
from bspytasks.benchmarks.vcdim.data_mgr import VCDimDataManager
from bspytasks.tasks.boolean.gate_finder import BooleanGateTask


class VCDimensionTest():

    def __init__(self, configs, excel_file):
        self.algorithm_configs = configs['algorithm_configs']
        # self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.output_dir = configs['results_base_dir']
        self.test_data_plot_name = '_plot.png'
        self.excel_file = excel_file
        self.threshold_parameter = configs['threshold_parameter']
        self.show_plots = configs['show_plots']
        self.amplitude_lengths = configs['algorithm_configs']['processor']['waveform']['amplitude_lengths']
        self.slope_lengths = configs['algorithm_configs']['processor']['waveform']['slope_lengths']

        self.data_manager = VCDimDataManager(configs)

    def init_test(self, vc_dimension):
        self.vc_dimension = vc_dimension
        self.threshold = self.calculate_threshold()
        self.readable_inputs, self.transformed_inputs, readable_targets, transformed_targets, found, self.mask = self.data_manager.get_data(vc_dimension)
        # if self.algorithm_configs['algorithm'] == 'gradient_descent' and self.algorithm_configs['processor']['platform'] == 'simulation':
        #    self.mask = None
        # else:
        #    self.mask = mask
        self.algorithm_configs['processor']['shape'] = self.transformed_inputs.shape[0]
        self.boolean_gate_task = BooleanGateTask(self.algorithm_configs)
        self.init_excel_file(readable_targets)
        self.excel_file.insert_column('label', readable_targets)
        self.excel_file.insert_column('encoded_label', transformed_targets)
        self.excel_file.insert_column('found', found)

    def init_excel_file(self, readable_targets):
        column_names = ['label', 'found', 'accuracy', 'best_output', 'control_voltages', 'correlation', 'best_performance', 'encoded_label']
        self.excel_file.init_data(column_names, readable_targets)
        self.excel_file.reset()

    def calculate_threshold(self):
        return 1 - (self.threshold_parameter / self.vc_dimension)

    def run_test(self):

        data = self.excel_file.data.loc[self.excel_file.data['found'] == False]  # noqa: E712
        print('---------------------------------------------')
        for _, row in data.iterrows():
            print(" Finding Label: " + str(row['label']))
            if self.find_label(row['label'], row['encoded_label']):
                print(' Label found.')
            else:
                print(' Label NOT found.')
            print('---------------------------------------------')

    def find_label(self, label, encoded_label):
        excel_results = self.boolean_gate_task.find_label(self.transformed_inputs, label, encoded_label, self.mask, self.threshold)
        self.excel_file.add_result(excel_results, label)
        return excel_results['found']

    def get_not_found_gates(self):
        return self.excel_file.data['label'].loc[self.excel_file.data['found'] == False].size  # noqa: E712

    def close_test(self):
        aux = self.excel_file.data.copy()
        aux.index = range(len(aux.index))
        tab_name = 'VC Dimension ' + str(self.vc_dimension) + ' Threshold ' + str(self.threshold)
        self.excel_file.save_tab(tab_name, data=aux)
        self.save_plots()
        self.close_algorithm()
        return self.oracle()

    def close_algorithm(self):
        try:
            self.algorithm.close()
        except AttributeError:
            print('There is no closing function for the current algorithm configuration. Skipping.')

    def save_plots(self):  # pylint: disable=E0202
        self.plot_results()
        for _, row in self.excel_file.data.iterrows():
            if len(np.unique(row['label'])) != 1:
                self.plot_output(row)

    def plot_results(self):
        plt.figure()
        fitness_classifier = self.excel_file.data['best_performance'].to_numpy()
        plt.plot(fitness_classifier, self.excel_file.data['accuracy'].to_numpy(), 'o')
        plt.plot(np.linspace(np.nanmin(fitness_classifier),
                             np.nanmax(fitness_classifier)),
                 self.threshold * np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness / Performance')
        plt.ylabel('Accuracy')
        path = self.output_dir + 'dimension_' + str(self.vc_dimension)
        create_directory(path)
        plt.savefig(os.path.join(path, self.test_data_plot_name))
        if self.show_plots:
            plt.show()

    def oracle(self):
        return self.excel_file.data.loc[self.excel_file.data['found'] == False].size == 0  # noqa: E712

    def plot_output(self, row):
        path = os.path.join(self.output_dir + 'dimension_' + str(self.vc_dimension), self.boolean_gate_task.is_found(row['found']))
        create_directory(path)
        self.boolean_gate_task.plot_gate(row, self.mask, self.show_plots, os.path.join(path, str(row['label']) + '_' + self.test_data_plot_name))

    # def format_genes(self):
    #     for i in range(len(self.genes_classifier)):
    #         if self.genes_classifier[i] is np.nan:
    #             self.genes_classifier[i] = np.nan * np.ones_like(self.genes_classifier[1])
    #             self.output_classifier[i] = np.nan * np.ones_like(self.output_classifier[1])

        # try:
        # not_found = self.found_classifier == 0
        # print('Classifiers not found: %s' %
        #       np.arange(len(self.found_classifier))[not_found])
        # binaries_nf = np.array(binary_labels)[not_found]  # labels not found
        # print('belongs to : \n', binaries_nf)
        # output_nf = self.output_classifier[not_found]
        # # plt output of failed classifiers
        # plt.figure()
        # plt.plot(output_nf.T)
        # plt.legend(binaries_nf)
        # # plt gnes with failed classifiers
        # plt.figure()
        # plt.hist(self.genes_classifier[not_found, :5], 30)
        # plt.legend([1, 2, 3, 4, 5])
        # plt.show()

        # except Exception:
        #     @todo improve the exception management
        #     print('Error in plotting output!')
