from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.ring_classification import RingClassificationTask as Task
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from bspytasks.utils.excel import get_series_with_numpy

RUNS = 4

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

plt.figure()
plt.plot(best_run['best_output'])
plt.show()
