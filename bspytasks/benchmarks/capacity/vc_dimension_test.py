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
from bspyalgo.algorithm_manager import get_algorithm
from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.io import create_directory
from bspyalgo.utils.performance import perceptron, corr_coeff
from bspytasks.benchmarks.capacity.interface import VCDimDataManager


class VCDimensionTest():

    def __init__(self, configs, excel_file):
        self.algorithm_configs = configs['algorithm_configs']
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.output_dir = configs['results_base_dir']
        self.test_data_plot_name = '_plot.png'
        self.excel_file = excel_file
        self.threshold_parameter = configs['threshold_parameter']
        self.show_plots = configs['show_plots']
        self.amplitude_lengths = configs['encoder']['waveform']['amplitude_lengths']
        self.slope_lengths = configs['encoder']['waveform']['slope_lengths']
        if self.algorithm_configs['algorithm'] == 'gradient_descent' and self.algorithm_configs['processor']['platform'] == 'simulation':
            self.find_label_core = self.find_label_with_torch
            self.ignore_label = self.ignore_label_with_torch
        else:
            self.find_label_core = self.find_label_with_numpy
            self.ignore_label = self.ignore_label_with_numpy
        self.data_manager = VCDimDataManager(configs)

    def init_test(self, vc_dimension):
        self.vc_dimension = vc_dimension
        self.threshold = self.calculate_threshold()
        self.readable_inputs, self.transformed_inputs, readable_targets, transformed_targets, found, self.mask = self.data_manager.get_data(vc_dimension)
        # if self.algorithm_configs['algorithm'] == 'gradient_descent' and self.algorithm_configs['processor']['platform'] == 'simulation':
        #    self.mask = None
        # else:
        #    self.mask = mask

        self.init_excel_file(readable_targets)
        self.excel_file.insert_column('label', readable_targets)
        self.excel_file.insert_column('encoded_label', transformed_targets)
        self.excel_file.insert_column('found', found)

    def init_excel_file(self, readable_targets):
        column_names = ['label', 'found', 'accuracy', 'best_output', 'control_voltages', 'correlation', 'best_performance', 'encoded_label']
        self.excel_file.init_data(readable_targets, column_names)

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

    def get_not_found_gates(self):
        return self.excel_file.data['label'].loc[self.excel_file.data['found'] == False].size  # noqa: E712

    def find_label(self, label, encoded_label):
        if len(np.unique(label)) == 1:
            print('Label ', label, ' ignored')
            excel_results = self.ignore_label(encoded_label)
        else:
            excel_results = self.find_label_core(encoded_label)
            excel_results['found'] = excel_results['accuracy'] >= self.threshold

        excel_results['label'] = label
        self.excel_file.add_result(label, excel_results)
        return excel_results['found']

    def optimize(self, encoded_label):
        algorithm_data = self.algorithm.optimize(self.transformed_inputs, encoded_label, mask=self.mask)
        algorithm_data.judge()
        excel_results = algorithm_data.results

        return excel_results

    def find_label_with_numpy(self, encoded_label):
        excel_results = self.optimize(encoded_label)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], encoded_label[excel_results['mask']])
        excel_results['encoded_label'] = encoded_label
        return excel_results

    def find_label_with_torch(self, encoded_label):
        encoded_label = TorchUtils.format_tensor(encoded_label)
        excel_results = self.optimize(encoded_label)
        excel_results['accuracy'], _, _ = perceptron(excel_results['best_output'][excel_results['mask']], TorchUtils.get_numpy_from_tensor(encoded_label[excel_results['mask']]))
        excel_results['encoded_label'] = encoded_label.cpu()
        # excel_results['targets'] = excel_results
        excel_results['correlation'] = corr_coeff(excel_results['best_output'][excel_results['mask']].T, excel_results['targets'].cpu()[excel_results['mask']].T)
        return excel_results

    def ignore_label_with_torch(self, encoded_label):
        excel_results = self.ignore_label_core()
        excel_results['encoded_label'] = encoded_label.cpu()
        return excel_results

    def ignore_label_with_numpy(self, encoded_label):
        excel_results = self.ignore_label_core()
        excel_results['encoded_label'] = encoded_label
        return excel_results

    def ignore_label_core(self):
        excel_results = {}
        excel_results['control_voltages'] = np.nan
        excel_results['best_output'] = np.nan
        excel_results['best_performance'] = np.nan
        excel_results['accuracy'] = np.nan
        excel_results['correlation'] = np.nan
        excel_results['found'] = True
        return excel_results

    def close_test(self):
        aux = self.excel_file.data.copy()
        aux.index = range(len(aux.index))
        tab_name = 'VC Dimension ' + str(self.vc_dimension) + ' Threshold ' + str(self.threshold)
        self.excel_file.save_tab(tab_name, data=aux)
        self.save_plots()
        return self.oracle()

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

    def plot_output(self, row):
        plt.figure()
        plt.plot(row['best_output'][self.mask])
        plt.plot(row['encoded_label'][self.mask])
        plt.xlabel('Current (nA)')
        plt.ylabel('Time')
        path = os.path.join(self.output_dir + 'dimension_' + str(self.vc_dimension), self.is_found(row['found']))
        create_directory(path)
        plt.savefig(os.path.join(path, str(row['label']) + '_' + self.test_data_plot_name))
        if self.show_plots:
            plt.show()

    def is_found(self, found):
        if found:
            return 'FOUND'
        else:
            return 'NOT_FOUND'

    def oracle(self):
        return self.excel_file.data.loc[self.excel_file.data['found'] == False].size == 0  # noqa: E712

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
