# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:10:31 2020

@author: Jochem
"""

import bspytasks.validation.gradient_ranking.gradient_utils as grad
import bspytasks.validation.electrode_ranking.perturbation_utils as pert

# Class for calculations
class GradientCalculator():
    def __init__(self, configs):
        self.model_data_path = configs['processor']['torch_model_dict']
        self.grid_mode = configs['gradient']['grid']['grid_mode']
        self.n_points = configs['gradient']['grid']['n_points']

    def get_averaged_values(self):
        self.grid = grad.create_grid_automatic(self.model_data_path, grid_mode=self.grid_mode, n_points=self.n_points)
        # grid = create_grid_manual(min_voltage, max_voltage, grid_mode = grid_mode, n_points = n_points) # Another option
        self.outputs, self.gradients = grad.get_outputs_gradients(self.model_data_path, self.grid, grid_mode=self.grid_mode)
        self.avg_grad = grad.averager(self.gradients, self.grid_mode)
        return self.avg_grad

    def rank_values(self):
        self.electrodes_sets = [[0], [1], [2], [3], [4], [5], [6]]
        self.ranked_descriptions, self.ranked_values, self.ranking_indices = pert.rank_low_to_high(self.electrodes_sets, self.avg_grad)
        return self.ranked_descriptions, self.ranked_values, self.ranking_indices
#%% For testing the code
if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    configs = load_configs('configs/benchmark_tests/analysis/gradient_analysis.json')
    calculator = GradientCalculator(configs)
    avg_grad = calculator.get_averaged_values()
    ranked_descriptions, ranked_values, ranking_indices = calculator.rank_values()