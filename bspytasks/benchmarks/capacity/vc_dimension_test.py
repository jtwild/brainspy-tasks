#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:18:29 2018
Wrapper to measure the VC dimension of a device using the measurement script measure_VCdim.py
This wrapper creates the binary labels for N points and for each label it finds the control voltages.
If successful (measured by a threshold on the correlation and by the perceptron accuracy), the entry 1 is set in a vector corresponding to all labellings.
@author: hruiz and ualegre
"""

import numpy as np
from matplotlib import pyplot as plt
from bspyalgo.algorithm_manager import get_algorithm
from bspytasks.benchmarks.capacity.interface import VCDimTargetEncoder, VCDimInputEncoder
import bspyinstr.utils.waveform as waveform
from bspytasks.utils.accuracy import perceptron


class VCDimensionTest():

    def __init__(self, configs, excel_file):
        self.algorithm = get_algorithm(configs['algorithm_configs'])
        self.output_dir = configs['results_base_dir']
        self.test_data_plot_name = '_plot.png'
        self.excel_file = excel_file
        # self.targets_encoder = VCDimTargetEncoder(configs['encoder']['waveform'])
        # self.inputs_encoder = VCDimInputEncoder(configs['encoder']['waveform'])
        self.threshold_parameter = configs['threshold_parameter']
        self.show_plots = configs['show_plots']
        self.amplitude_lengths = configs['encoder']['waveform']['amplitude_lengths']
        self.slope_lengths = configs['encoder']['waveform']['slope_lengths']
        # if configs['algorithm_configs']['algorithm'] == 'genetic':
        #     self.accuracy = perceptron
        # elif configs['algorithm_configs']['algorithm'] == 'gradient_descent':

    def generate_test_inputs(self, vc_dimension):
        # @todo create a function that automatically generates non-linear inputs
        # try:
        if vc_dimension == 4:
            return [[-0.7, -0.7, 0.7, 0.7], [0.7, -0.7, 0.7, -0.7]]  # Dimensions x Samples
        elif vc_dimension == 5:
            return [[-0.7, -0.7, 0.7, 0.7, -0.35],
                    [0.7, -0.7, 0.7, -0.7, 0.0]]
        elif vc_dimension == 6:
            return [[-0.7, -0.7, 0.7, 0.7, -0.35, 0.35],
                    [0.7, -0.7, 0.7, -0.7, 0.0, 0.0]]
        elif vc_dimension == 7:
            return [[-0.7, -0.7, 0.7, 0.7, -0.35, 0.35, 0.0],
                    [0.7, -0.7, 0.7, -0.7, 0.0, 0.0, 1.0]]
        elif vc_dimension == 8:
            return [[-0.7, -0.7, 0.7, 0.7, -0.35, 0.35, 0.0, 0.0],
                    [0.7, -0.7, 0.7, -0.7, 0.0, 0.0, 1.0, -1.0]]
        # else:
            #         raise VCDimensionException()
            # except VCDimensionException:
            # print('Dimension Exception occurred. The selected VC Dimension is %d Please insert a value between ' % vc_dimension)

    def generate_binary_labels(self, vc_dimension):
        # length of list, i.e. number of binary targets
        binary_target_no = 2**vc_dimension
        assignments = []
        list_buf = []

        # construct assignments per element i
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
        print(binary_targets)
        print('===' * vc_dimension)
        return binary_targets

    def init_test(self, vc_dimension):
        self.vc_dimension = vc_dimension
        self.threshold = self.calculate_threshold()
        self.inputs = self.generate_test_inputs(vc_dimension)
        self.binary_labels = self.generate_binary_labels(vc_dimension)
        # self.targets_encoder.load_encoders(vc_dimension)
        # self.inputs_encoder.load_encoders(vc_dimension)

        self.init_excel_file()

    def init_excel_file(self):
        column_names = ['gate', 'found', 'accuracy', 'best_output', 'control_voltages', 'correlation', 'best_performance']
        self.excel_file.init_data(self.binary_labels, column_names)

    def calculate_threshold(self):
        return 1 - (self.threshold_parameter / self.vc_dimension)

    def run_test(self, binary_labels=np.array([])):
        if binary_labels.size == 0:
            binary_labels = self.binary_labels
        for label in binary_labels:
            self.find_label(label)

    def get_not_found_gates(self):
        return self.excel_file.data['gate'].loc[self.excel_file.data['found'] == False]  # noqa: E712

    def find_label(self, label):

        if len(np.unique(label)) == 1:
            print('Label ', label, ' ignored')
            excel_results = self.ignore_label({})
        else:
            print('Finding classifier ', label)
            # [:, :, np.newaxis]
            inputs_waveform = np.array([])
            for inp in self.inputs:
                input_waveform = waveform.generate_waveform(inp, self.amplitude_lengths, slope_lengths=self.slope_lengths)
                if inputs_waveform.shape == (0,):
                    inputs_waveform = np.concatenate((inputs_waveform, input_waveform))
                else:
                    inputs_waveform = np.vstack((inputs_waveform, input_waveform))
            # if len(inputs_waveform.shape) == 1:
            #    inputs_waveform = inputs_waveform[:, np.newaxis]
            inputs_waveform = inputs_waveform.T  # device_model --> (samples,dimension) ; device --> (dimensions,samples)
            target_waveform = waveform.generate_waveform(label, self.amplitude_lengths, slope_lengths=self.slope_lengths)
            target_waveform = target_waveform[:, np.newaxis]
            mask = waveform.generate_mask(label, self.amplitude_lengths, slope_lengths=self.slope_lengths)

            algorithm_data = self.algorithm.optimize(inputs_waveform, target_waveform, mask=mask)
            # algorithm_data = self.algorithm.optimize(self.inputs_encoder.inputs, label)

            algorithm_data.judge()
            excel_results = algorithm_data.results
            excel_results['accuracy'], _, _ = perceptron(algorithm_data.results['best_output'][mask], target_waveform[mask])
            excel_results['found'] = excel_results['accuracy'] >= self.threshold

        excel_results['gate'] = label
        self.excel_file.add_result(label, excel_results)
        # column_names = ['gate', 'found', 'accuracy', 'best_output', 'control_voltages', 'correlation', 'best_performance']
        # GA tiene un stopping criteria
        # Se define a traves de la correlacion
        # La correlacion tiene que ser super alta debe ser 95

        # row = {'gate': label, 'found': found}

    def ignore_label(self, excel_results):
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
        self.save_plot()
        return self.oracle()

    def save_plot(self):  # pylint: disable=E0202
        plt.figure()
        fitness_classifier = self.excel_file.data['best_performance'].to_numpy()
        plt.plot(fitness_classifier, self.excel_file.data['accuracy'].to_numpy(), 'o')
        plt.plot(np.linspace(np.nanmin(fitness_classifier),
                             np.nanmax(fitness_classifier)),
                 self.threshold * np.ones_like(np.linspace(0, 1)), '-k')
        plt.xlabel('Fitness')
        plt.ylabel('Accuracy')
        plt.savefig(self.output_dir + 'dimension_' + str(self.vc_dimension) + self.test_data_plot_name)
        if self.show_plots:
            plt.show()

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
