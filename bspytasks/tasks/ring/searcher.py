
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from bspyproc.utils.pytorch import TorchUtils
from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.classifier import RingClassificationTask as Task
RUNS = 10

configs = load_configs('configs/tasks/ring/template_gd_architecture_3.json')
task = Task(configs)


performance_per_run = np.zeros(RUNS)
correlation_per_run = np.zeros(RUNS)
accuracy_per_run = np.zeros(RUNS)
seeds_per_run = np.zeros(RUNS)
best_run = None

for run in range(RUNS):
    print(f'########### RUN {run} ################')
    seed = TorchUtils.init_seed(None, deterministic=True)
    excel_results = task.run_task(run=run)
    excel_results['seed'] = seed
    accuracy_per_run[run] = excel_results['accuracy']
    performance_per_run[run] = excel_results["best_performance"]
    correlation_per_run[run] = excel_results["correlation"]
    if best_run == None or best_run['best_performance'] < excel_results['best_performance']:
        excel_results['index'] = run
        best_run = excel_results
        task.close_test(run)
        pickle.dump(best_run, open("best_output.p", "wb"))

np.savez('general_info', performance=performance_per_run, correlation=correlation_per_run, accuracy=accuracy_per_run)
best_index = best_run['index']
best_run_output = best_run['best_output']  # best_output_run[best_index]
performance = best_run["best_performance"]  # performance_per_run[best_index]


print(f"Best performance {performance} in run {best_index} with corr. {correlation_per_run[best_index]}")
np.savez(os.path.join(task.configs["results_base_dir"], f'output_best_of_{RUNS}.npz'),
         index=best_index, best_run=best_run_output, performance=performance)

plt.figure()
plt.plot(correlation_per_run, performance_per_run, 'o')
plt.title('Correlation vs Fisher')
plt.xlabel('Correlation')
plt.ylabel('Fisher value')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'correlation_vs_fisher'))

plt.figure()
plt.plot(best_run_output)
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

plt.figure()
plt.hist(accuracy_per_run, 100)
plt.title('Histogram of Accuracy values')
plt.xlabel('Accuracy values')
plt.ylabel('Counts')
plt.savefig(os.path.join(task.configs["results_base_dir"], 'accuracy_histogram'))

plt.show()
