#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz and ualegre
"""
from bspytasks.benchmarks.capacity.vc_dimension_test import VCDimensionTest
from bspytasks.utils.excel import ExcelFile
from bspyalgo.utils.io import save, load_configs

import numpy as np


class CapacityTest():

    def __init__(self, configs):
        self.configs = configs
        self.current_dimension = configs['from_dimension']
        configs['algorithm_configs']['results_base_dir'] = configs['results_base_dir']
        self.excel_file = ExcelFile(configs['results_base_dir'] + 'capacity_test_results.xlsx')
        self.vcdimension_test = VCDimensionTest(configs, self.excel_file)

    def run_test(self):
        results = {}

        while True:
            print('==== VC Dimension %d ====' % self.current_dimension)
            self.vcdimension_test.init_test(self.current_dimension)
            opportunity = 0
            not_found = np.array([])
            while True:
                self.vcdimension_test.run_test(binary_labels=not_found)
                not_found = self.vcdimension_test.get_not_found_gates()
                opportunity += 1
                if (not_found.size == 0) or (opportunity >= self.configs['max_opportunities']):
                    break
            results[str(self.current_dimension)] = self.vcdimension_test.close_test()
            if not results[str(self.current_dimension)] or not self.next_vcdimension():
                return self.close_test(results)

    def close_test(self, results):
        self.excel_file.save_file()
        save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', data=self.configs)
        self.results = results
        return results

    def next_vcdimension(self):
        if self.current_dimension + 1 > self.configs['to_dimension']:
            return False
        else:
            self.current_dimension += 1
            return True


if __name__ == '__main__':

    # platform = {}
    # platform['modality'] = 'simulation_nn'
    # # platform['path2NN'] = r'D:\UTWENTE\PROJECTS\DARWIN\Data\Mark\MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    # platform['path2NN'] = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/checkpoint3000_02-07-23h47m.pt'
    # # platform['path2NN'] = r'/home/hruiz/Documents/PROJECTS/DARWIN/Data_Darwin/Devices/Marks_Data/April_2019/MSE_n_d10w90_200ep_lr1e-3_b1024_b1b2_0.90.75.pt'
    # platform['amplification'] = 10.
    # ga_configs = {}
    # ga_configs['partition'] = [5] * 5  # Partitions of population
    # # Voltage range of CVs in V
    # ga_configs['generange'] = [[-1.2, 0.6], [-1.2, 0.6],
    #                            [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3], [1, 1]]
    # ga_configs['genes'] = len(ga_configs['generange'])    # Nr of genes
    # # Nr of individuals in population
    # ga_configs['genomes'] = sum(ga_configs['partition'])
    # ga_configs['mutationrate'] = 0.1

    # # Parameters to define target waveforms
    # ga_configs['lengths'] = [80]     # Length of data in the waveform
    # # Length of ramping from one value to the next
    # ga_configs['slopes'] = [0]  # Parameters to define task
    # ga_configs['fitness'] = 'corrsig_fit'  # 'corr_fit'
    # ga_configs['platform'] = platform  # Dictionary containing all variables for the platform

    # capacity_test_configs = {}
    # capacity_test_configs['output_dir'] = r'/home/unai/Documents/3-programming/boron-doped-silicon-chip-simulation/'
    # capacity_test_configs['surrogate_model_name'] = 'checkpoint3000_02-07-23h47m'
    # capacity_test_configs['from_dimension'] = 4
    # capacity_test_configs['to_dimension'] = 5
    # capacity_test_configs['max_opportunities'] = 3
    # capacity_test_configs['threshold_parameter'] = 0.5
    # capacity_test_configs['show_plot'] = False
    capacity_test_configs = load_configs('configs/benchmark_tests/capacity_test/capacity_test_template_gd.json')

    test = CapacityTest(capacity_test_configs)
    test.run_test()
