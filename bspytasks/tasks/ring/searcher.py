from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.ring_classification import RingClassificationTask as Task
import os
import numpy as np
import matplotlib.pyplot as plt

RUNS = 10

task = Task(load_configs('configs/tasks/ring/template_gd_architecture_2.json'))

performance_per_run = np.zeros(RUNS)
correlation_per_run = np.zeros(RUNS)
best_output_run = []

for run in range(RUNS):
    print(f'########### RUN {run} ################')
    excel_results, _ = task.run_task(run=run)
    performance_per_run[run] = excel_results["best_performance"]
    correlation_per_run[run] = excel_results["correlation"]
    best_output_run.append(excel_results['best_output'])


best_index = np.argmin(performance_per_run)
best_run = best_output_run[best_index]
performance = performance_per_run[best_index]
print(f"Best performance {performance} in run {best_index} with corr. {correlation_per_run[best_index]}")
np.savez(os.path.join(task.configs["results_base_dir"], f'output_best_of_{RUNS}.npz'),
         index=best_index, best_run=best_run, performance=performance)

plt.figure()
plt.plot(correlation_per_run, performance_per_run, 'o')
plt.title('Correlation vs Fisher')
plt.xlabel('Correlation')
plt.ylabel('Fisher value')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'correlation_vs_fisher'))

plt.figure()
plt.plot(best_run)
plt.title(f'Best Output, run:{best_index}, F-val: {performance}')
plt.xlabel('Time points (a.u.)')
plt.ylabel('Output current (nA)')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'output_best_run'))

plt.figure()
plt.hist(performance_per_run, 100)
plt.title('Histogram of Fisher values')
plt.xlabel('Fisher values')
plt.ylabel('Counts')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'fisher_values_histogram'))

plt.show()
