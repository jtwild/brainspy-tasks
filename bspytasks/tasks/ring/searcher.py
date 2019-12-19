from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.ring_classification import RingClassificationTask as Task
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from bspytasks.utils.excel import get_series_with_numpy

RUNS = 1000

task = Task(load_configs('configs/tasks/ring/template_gd_architecture.json'))
for run in range(RUNS):
    result = task.run_task(run=run)
task.close_test()

excel_results = pd.read_pickle(os.path.join(task.configs["results_base_dir"], 'results.pkl'))


performance_per_run = excel_results["best_performance"]
correlation_per_run = excel_results["correlation"]

best_index = performance_per_run.astype(float).idxmin()
best_run = excel_results.iloc[best_index]
print(f"Best performance {best_run['best_performance']} in run {best_index} with corr. {best_run['correlation']}")

plt.figure()
plt.plot(correlation_per_run.to_numpy(), performance_per_run.to_numpy(), '.')
plt.title('Correlation vs Fisher')
plt.xlabel('Correlation')
plt.ylabel('Fisher value')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'correlation_vs_fisher'))

plt.figure()
plt.plot(best_run['best_output'])
plt.title('Best Output')
plt.xlabel('Time points (a.u.)')
plt.ylabel('Output current (nA)')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'output_best_run'))

plt.figure()
plt.hist(performance_per_run.astype(float).to_numpy())
plt.title('Histogram of Fisher values')
plt.xlabel('Fisher values')
plt.ylabel('Counts')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'fisher_values_histogram'))

plt.show()
