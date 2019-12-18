from bspyalgo.utils.io import load_configs
from bspytasks.tasks.ring.ring_classification import RingClassificationTask as Task
import pandas as pd
import os
import numpy as np
from bspytasks.utils.excel import get_series_with_numpy

RUNS = 100

task = Task(load_configs('configs/tasks/ring/template_gd_architecture.json'))
for run in range(RUNS):
    result = task.run_task(run=run)
    task.close_test()


excel_results = pd.read_excel(os.path.join(task.configs['results_base_dir'], "experiment_results.xlsx"))

performance_per_run = get_series_with_numpy(excel_results["best_performance"])
correlation_per_run = get_series_with_numpy(excel_results["correlation"])
best_index = performance_per_run.idxmin()
best_corr = correlation_per_run[best_index]
print(f"Best performance {performance_per_run.min()} in run {best_index} with corr. {}")
