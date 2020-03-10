# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 23:25:00 2020

@author: Jochem
"""
import pickle
file_path = r'C:\Users\Jochem\STACK\Daily_Usage\GIT_SourceTree\brainspy-tasks\tmp\output\filter_finder\march_single\succes_config\SUCCES_RUN2\patch_filter_4_points_2020_03_10_193822.47\gradient_descent_data_2020_03_10_194447.48\reproducibility\results.pickle'
with open(file_path, 'rb') as input_file:
    results = pickle.load(input_file)

controls = results['processor'].get_control_voltages()
inputs = results['processor'].input
outputs = results['processor'].output
processor = results['processor']
