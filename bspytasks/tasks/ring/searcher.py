
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.classifier import RingClassificationTask as Task
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspytasks.tasks.ring.plotter import ArchitecturePlotter

from bspyalgo.utils.io import create_directory, create_directory_timestamp


class RingSearcher():

    def __init__(self, configs, is_main=True):
        self.configs = configs
        self.is_main = is_main
        self.base_dir = configs['results_base_dir']
        self.task = Task(configs, is_main=False)
        self.data_loader = RingDataLoader(configs)
        self.plotter = ArchitecturePlotter(configs)

    def init_dirs(self, gap):
        main_dir = f'searcher_{gap}mV'
        search_stats_dir = 'search_stats'

        if self.is_main:
            base_dir = create_directory_timestamp(self.base_dir, main_dir)
        else:
            base_dir = os.path.join(self.base_dir, main_dir)
            create_directory(base_dir)
        self.search_stats_dir = os.path.join(base_dir, search_stats_dir)
        create_directory(self.search_stats_dir)

        self.task.init_dirs(base_dir)
        return configs

    def reset(self):
        self.task.reset()
        self.performance_per_run = np.zeros(self.configs['runs'])
        self.correlation_per_run = np.zeros(self.configs['runs'])
        self.accuracy_per_run = np.zeros(self.configs['runs'])
        self.best_run = None
        # os.mkdir(os.path.join(self.configs["results_base_dir"], 'reproducibility'))
        # os.mkdir(os.path.join(self.configs["results_base_dir"], 'search_stats'))
        # os.mkdir(os.path.join(self.configs["results_base_dir"], 'results'))

    def search_solution(self, gap):
        self.init_dirs(gap)
        self.reset()
        self.task.configs['ring_data']['gap'] = gap
        inputs, targets, mask = self.data_loader.generate_new_data(self.configs['algorithm_configs']['processor'], gap=gap)
        for run in range(self.configs['runs']):
            print(f'########### RUN {run} ################')
            seed = TorchUtils.init_seed(None, deterministic=True)
            results = self.task.run_task(inputs, targets, mask)
            results['seed'] = seed
            self.update_search_stats(results, run)
            if self.best_run == None or results['best_performance'] < self.best_run['best_performance']:
                self.update_best_run(results, run)
                self.task.save_plots(self.best_run, inputs, targets, mask, show_plot=self.configs["show_plots"], run=run)

        self.close_search()

    def update_search_stats(self, results, run):
        self.accuracy_per_run[run] = results['accuracy']
        self.performance_per_run[run] = results["best_performance"]
        self.correlation_per_run[run] = results["correlation"]

    def update_best_run(self, results, run):
        results['index'] = run
        del results['processor']
        self.best_run = results
        self.task.save_reproducibility_data(results)

    def close_search(self):
        np.savez(os.path.join(self.search_stats_dir, f"search_data_{self.configs['runs']}_runs.npz"), performance=self.performance_per_run, correlation=self.correlation_per_run, accuracy=self.accuracy_per_run)
        self.plot_search_results()

    def plot_search_results(self):
        best_index = self.best_run['index']
        best_run_output = self.best_run['best_output']
        performance = self.best_run["best_performance"]
        print(f"Best performance {performance} in run {best_index} with corr. {self.correlation_per_run[best_index]}")

        # plt.figure()
        # plt.plot(best_run_output)
        # plt.title(f'Best Output, run:{best_index}, F-val: {performance}')
        # plt.xlabel('Time points (a.u.)')
        # plt.ylabel('Output current (nA)')
        # plt.savefig(os.path.join(self.results_dir, f'best_output.eps'))

        plt.figure()
        plt.plot(self.correlation_per_run, self.performance_per_run, 'o')
        plt.title('Correlation vs Fisher')
        plt.xlabel('Correlation')
        plt.ylabel('Fisher value')
        plt.savefig(os.path.join(self.search_stats_dir, 'correlation_vs_fisher.eps'))

        plt.figure()
        plt.hist(self.performance_per_run, 100)
        plt.title('Histogram of Fisher values')
        plt.xlabel('Fisher values')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(self.search_stats_dir, 'fisher_values_histogram.eps'))

        plt.figure()
        plt.hist(self.accuracy_per_run, 100)
        plt.title('Histogram of Accuracy values')
        plt.xlabel('Accuracy values')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(self.search_stats_dir, 'accuracy_histogram'))

        if self.configs["show_plots"]:
            plt.show()


if __name__ == '__main__':
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs

    # configs = load_configs('configs/tasks/ring/template_gd_architecture_cdaq_to_nidaq_validation2.json')
    configs = load_configs('configs/tasks/ring/template_gd_architecture_3.json')
    searcher = RingSearcher(configs)
    searcher.search_solution(0.2)
