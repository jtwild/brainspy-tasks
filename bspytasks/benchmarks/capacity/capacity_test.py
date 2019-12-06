#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:32:25 2018
This script generates all binary assignments of N elements.
@author: hruiz and ualegre
"""
from bspytasks.benchmarks.vcdim.vc_dimension_test import VCDimensionTest
from bspyalgo.utils.io import save, load_configs


class CapacityTest():

    def __init__(self, configs):
        self.configs = configs
        self.current_dimension = configs['from_dimension']
        self.vcdimension_test = VCDimensionTest(configs['vc_dimension_test'])

    def save_configs(self, configs):
        results_dir = configs['vc_dimension_test']['boolean_gate_test']['results_dir']
        overwrite = configs['vc_dimension_test']['boolean_gate_test']['overwrite']
        save(mode='configs', path=results_dir, filename='test_configs.json', overwrite=overwrite, data=configs)

    def run_test(self):
        # results = {}
        print('*****************************************************************************************')
        print(f"CAPACITY TEST FROM VCDIM {self.configs['from_dimension']} TO VCDIM {self.configs['to_dimension']} ")
        print('*****************************************************************************************')
        self.save_configs(self.configs)
        while True:
            self.vcdimension_test.run_test(self.current_dimension)
            if not self.next_vcdimension():
                break
        # while True:
        # print('==== VC Dimension %d ====' % self.current_dimension)
        # self.vcdimension_test.init_test(self.current_dimension)
        # opportunity = 0
        # not_found = np.array([])
        # while True:
        #    self.vcdimension_test.run_test(self.current_dimension)
        #     not_found = self.vcdimension_test.get_not_found_gates()
        #     opportunity += 1
        #     if (not_found < 1) or (opportunity >= self.configs['max_opportunities']):
        #         break
        # results[str(self.current_dimension)] = self.vcdimension_test.close_test()
        # if not self.next_vcdimension():  # not results[str(self.current_dimension)] or

        self.vcdimension_test.close_results_file()
        print('*****************************************************************************************')
    # def close_test(self, results):
    #     self.vcdimension_test.close_results_file()
    #     self.results = results
    #     return results

    def next_vcdimension(self):
        if self.current_dimension + 1 > self.configs['to_dimension']:
            return False
        else:
            self.current_dimension += 1
            return True


if __name__ == '__main__':
    configs = load_configs('configs/benchmark_tests/capacity/template_gd_architecture_221.json')

    test = CapacityTest(configs['capacity_test'])
    test.run_test()
