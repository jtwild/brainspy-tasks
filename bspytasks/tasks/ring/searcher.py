
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.classifier import RingClassificationTask as Task
from bspytasks.tasks.ring.data_loader import RingDataLoader
from bspytasks.tasks.ring.plotter import ArchitecturePlotter


class RingSearcher():

    def __init__(self, configs):
        self.configs = configs
        self.task = Task(configs)
        self.data_loader = RingDataLoader(configs)
        self.plotter = ArchitecturePlotter(configs)

    def reset(self):
        self.task.reset()
        self.performance_per_run = np.zeros(self.configs['runs'])
        self.correlation_per_run = np.zeros(self.configs['runs'])
        self.accuracy_per_run = np.zeros(self.configs['runs'])
        self.best_run = None
        os.mkdir(os.path.join(self.configs["results_base_dir"], 'reproducibility'))
        os.mkdir(os.path.join(self.configs["results_base_dir"], 'search_stats'))
        os.mkdir(os.path.join(self.configs["results_base_dir"], 'results'))

    def search_solution(self, gap):
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
                self.plotter.save_plots(self.best_run, inputs, targets, mask, self.configs, show_plot=self.configs["show_plots"], run=run)
                pickle.dump(self.best_run, open(os.path.join(os.path.join(self.configs["results_base_dir"], 'reproducibility'), f"best_output_results.pkl"), "wb"))

        self.close_search()

    def update_search_stats(self, results, run):
        self.accuracy_per_run[run] = results['accuracy']
        self.performance_per_run[run] = results["best_performance"]
        self.correlation_per_run[run] = results["correlation"]

    def update_best_run(self, results, run):
        results['index'] = run
        del results['processor']
        self.best_run = results
        self.task.close_test()

    def close_search(self):
        np.savez(os.path.join(os.path.join(self.configs["results_base_dir"], 'search_stats'), f"search_data_{self.configs['runs']}_runs.npz"), performance=self.performance_per_run, correlation=self.correlation_per_run, accuracy=self.accuracy_per_run)
        self.plot_search_results()

    def plot_search_results(self):
        best_index = self.best_run['index']
        best_run_output = self.best_run['best_output']
        performance = self.best_run["best_performance"]
        print(f"Best performance {performance} in run {best_index} with corr. {self.correlation_per_run[best_index]}")

        plt.figure()
        plt.plot(best_run_output)
        plt.title(f'Best Output, run:{best_index}, F-val: {performance}')
        plt.xlabel('Time points (a.u.)')
        plt.ylabel('Output current (nA)')
        plt.savefig(os.path.join(os.path.join(self.configs["results_base_dir"], 'results'), f'best_output'))

        plt.figure()
        plt.plot(self.correlation_per_run, self.performance_per_run, 'o')
        plt.title('Correlation vs Fisher')
        plt.xlabel('Correlation')
        plt.ylabel('Fisher value')
        plt.savefig(os.path.join(os.path.join(self.configs["results_base_dir"], 'search_stats'), 'correlation_vs_fisher'))

        plt.figure()
        plt.hist(self.performance_per_run, 100)
        plt.title('Histogram of Fisher values')
        plt.xlabel('Fisher values')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(os.path.join(self.configs["results_base_dir"], 'search_stats'), 'fisher_values_histogram'))

        plt.figure()
        plt.hist(self.accuracy_per_run, 100)
        plt.title('Histogram of Accuracy values')
        plt.xlabel('Accuracy values')
        plt.ylabel('Counts')
        plt.savefig(os.path.join(os.path.join(self.configs["results_base_dir"], 'search_stats'), 'accuracy_histogram'))

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
    searcher.search_solution(0.39)
