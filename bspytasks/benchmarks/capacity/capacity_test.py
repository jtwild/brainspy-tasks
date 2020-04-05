#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz and ualegre
"""
import os
from bspytasks.benchmarks.vcdim.vc_dimension_test import VCDimensionTest
from bspyalgo.utils.io import create_directory_timestamp, save


class CapacityTest():

    def __init__(self, configs):
        self.configs = configs
        self.current_dimension = configs['from_dimension']
        configs = self.init_dirs(configs)
        self.vcdimension_test = VCDimensionTest(
            configs['vc_dimension_test'], is_main=False)

    def init_dirs(self, configs):
        base_dir = create_directory_timestamp(
            configs['results_base_dir'], 'capacity_test')
        configs['vc_dimension_test']['results_base_dir'] = base_dir
        self.configs_dir = os.path.join(base_dir, 'test_configs.json')
        return configs

    def run_test(self, validate=False):
        print('*****************************************************************************************')
        print(
            f"CAPACITY TEST FROM VCDIM {self.configs['from_dimension']} TO VCDIM {self.configs['to_dimension']} ")
        print('*****************************************************************************************')
        save(mode='configs', file_path=self.configs_dir, data=configs)
        while True:
            self.vcdimension_test.run_test(self.current_dimension, validate=validate)
            if not self.next_vcdimension():
                break

        self.vcdimension_test.close_results_file()
        print('*****************************************************************************************')

    def next_vcdimension(self):
        if self.current_dimension + 1 > self.configs['to_dimension']:
            return False
        else:
            self.current_dimension += 1
            return True


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs(
        'configs/benchmark_tests/capacity/template_ga_simulation.json')

    test = CapacityTest(configs['capacity_test'])
    test.run_test(validate=False)
