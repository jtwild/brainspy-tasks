# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:47:22 2020

@author: Jochem
"""
# %% Import packages
from bspytasks.benchmarks.capacity.capacity_test import CapacityTest
from bspyalgo.utils.io import load_configs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from bspyproc.processors.simulation.surrogate import SurrogateModel
# %% User data
base_configs = load_configs('configs/benchmark_tests/capacity/template_gd.json')
output_directory = "tmp/output/test/volt_loop/"
# Lists to loop over
input_elecs = np.array([0, 1, 2, 3, 4, 5, 6])
#torch_model_dict_list taken from model list given
voltage_intervals = 1/8*np.array([1, 2, 3, 4, 5, 6, 7, 8])
vcs = np.arange(base_configs['capacity_test']['from_dimension'], base_configs['capacity_test']['to_dimension']+1, step=1) # to_dimension done +1 because endpoints are not used by np.arange
vc_dims = np.array(['vc'+ str(vc) for vc in vcs]) # has the string 'vc8' for example, isntead of just 8.

# %% Auto load torch models from directory
base_dir = r"C:\Users\Jochem\STACK\Daily_Usage\Bestanden\UT\TN_MSc\Afstuderen\models\ordered_one_each"
glob_filter = '**/*.pt'
glob_query = os.path.join(base_dir, glob_filter)
# And finally find the models we want to use
torch_model_dicts = glob.glob(glob_query, recursive=True)
# Create dataframe to store results and voltage intervals
df_index = pd.MultiIndex.from_product([input_elecs, voltage_intervals, torch_model_dicts], names = ['input_elec','voltage_interval','model'])
df_columns = np.concatenate((vc_dims, ['volt_min','volt_max','capacities']))
df_capacity = pd.DataFrame(index = df_index, columns = df_columns)

# Fill minmax voltage intervals
for input_elec in input_elecs:
    for voltage_interval in voltage_intervals:
        for torch_model_dict in torch_model_dicts:
            processor = SurrogateModel({'torch_model_dict': torch_model_dict})
            min_voltage = (processor.offset[input_elec] - voltage_interval*processor.amplitude[input_elec]).item()
            max_voltage = (processor.offset[input_elec] + voltage_interval*processor.amplitude[input_elec]).item()
            # Store minmax values in dataframe
            df_filter = (input_elec, voltage_interval, torch_model_dict)
            df_capacity.loc[df_filter, 'volt_min'] = min_voltage
            df_capacity.loc[df_filter, 'volt_max'] = max_voltage

# %% Do the capacity test
base_configs['capacity_test']['results_base_dir'] = output_directory
for input_elec in input_elecs:
    for voltage_interval in voltage_intervals:
        for torch_model_dict in torch_model_dicts:
            configs = base_configs.copy()
            df_filter = (input_elec, voltage_interval, torch_model_dict)
            # Edit the configs
            configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['input_indices'] = [input_elec]
            configs['capacity_test']['vc_dimension_test']['boolean_gate_test']['algorithm_configs']['processor']['torch_model_dict'] = torch_model_dict
            configs['capacity_test']['vc_dimension_test']['voltage_intervals'] = [df_capacity.loc[df_filter, 'volt_min'], df_capacity.loc[df_filter, 'volt_max']]

            # Start the test
            test = CapacityTest(configs['capacity_test'])
            test.run_test(validate=False)
            plt.close('all')

            # Save some extra results

            df_capacity.loc[df_filter,'capacities'] = test.summary_results['capacity_per_N']
#            df_capacity.loc[df_filter,'summary'] = test.summary_results.items() # apparantly you cannot pickle dictionary itmems with pandas
#            for m, vc_dim in enumerate(vc_dims):
#                df_capacity.loc[df_filter, vc_dim] = test.summary_results['capacity_per_N'][m]

#%% Save loop items
df_capacity.to_pickle(os.path.join(output_directory, 'capacity_loop_data.pkl'))