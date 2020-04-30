# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:21:44 2020

@author: Jochem
"""
# Load packages
import numpy as np
import matplotlib.pyplot as plt
import bspytasks.validation.electrode_ranking.perturbation_utils as pert
from bspyalgo.utils.io import load_configs

# Class definition
class ElectrodeRanker():
    def __init__(self,configs):
        #  Load User variables
        self.configs = configs
        # Get config values
        self.electrodes_sets = configs['perturbation']['electrodes_sets']
        self.perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']

    def get_data(self):
        # Get unperturbed data
        self.inputs_unperturbed, self.targets_loaded, self.info = pert.load_data(configs)
        self.targets = pert.get_prediction(configs, self.inputs_unperturbed).flatten()

    def rank(self, sub_plot_type = 'ranking'):
        # Get data
        self.get_data()
        # Initialize values
        self.rmse = np.zeros((len(self.perturb_fraction_sets), len(self.electrodes_sets)))
        fig_hist, axs_hist = plt.subplots(2, 4)
        axs_hist = axs_hist.flatten()
        fig_bar, axs_bar = plt.subplots(2, 4)
        axs_bar = axs_bar.flatten()
        counter = 0
        # Start loop over config values
        for i in range(len(self.perturb_fraction_sets)):
            configs['perturbation']['perturb_fraction'] = self.perturb_fraction_sets[i]
            for j in range(len(self.electrodes_sets)):
                configs['perturbation']['electrodes'] = self.electrodes_sets[j]
                electrode = self.electrodes_sets[j][0]  # does not work if more than one electrode is perturbed..., so 'fixed' by taking only first component.
                # Perturb data, get prediciton, get error, get rmse
                inputs_perturbed = pert.perturb_data(self.configs, self.inputs_unperturbed)
                prediction = pert.get_prediction(self.configs, inputs_perturbed)
                # Real error
                error = prediction - self.targets  # for unkown sizes can use lists [([[]]*10)]*5 and convert to numpy afterwards
                # Make subsets for voltage ranges and rankn them
                error_subsets, grid, ranges = pert.sort_by_input_voltage(self.inputs_unperturbed[:, electrode], error,
                                                                         min_val=-1.2, max_val=0.6, granularity=0.1)
                pert.plot_hists(np.abs(error_subsets), ax=axs_hist[counter], legend=grid.round(2).tolist())
                pert.rank_low_to_high(grid,
                                      np.sqrt(pert.np_object_array_mean(error_subsets**2)),
                                      plot_type=sub_plot_type, ax=axs_bar[counter], x_data = grid)
                fig_bar.suptitle('Voltage range ranking based on RMSE on interval')
                # And root mean square error for the total electrode
                self.rmse[i, j] = np.sqrt(np.mean(error**2))
                counter += 1

    def plot_rank(self, plot_type = 'ranking'):
        # Ranking electrode importance
        pert.rank_low_to_high(self.electrodes_sets, self.rmse[0, :], plot_type=plot_type)
        plt.xlabel('Electrode #')
        plt.title('Electrode ranking based on RMSE')

#%% Test code:
if __name__ == '__main__':
    configs = load_configs('configs/validation/single_perturbation_all_electrodes_configs.json')
    ranker = ElectrodeRanker(configs)
    ranker.rank()
    ranker.plot_rank()