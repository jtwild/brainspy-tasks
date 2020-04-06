"""
First version on perturbed inputs:
 Noise ('perturbation') is added to the given electrodes, and then it is visualized how the RMSE changes. User variables are
- electrodes, which will be perturbed
- cofnig file which contains the model and corresponding measurement test data
- perturb_fraction which determines how much noise is added relative to the magntiude of the input voltage range. If the input voltages are within -1.2,+0.6V, then an perturb_fraction of 0.1 means 10% of this range is used, so 0.18V. This 0.18V is used to generate noise evenly distributed in the interval [-0.18/2, +0.18/2]. So in the interval [-0.09, +0.09]V.

Author: Jochem Wildeboer
"""
from bspyalgo.utils.io import load_configs
from bspysmg.model.data.outputs import test_model
from bspysmg.model.data.outputs.train_model import train_surrogate_model
import numpy as np
from bspyproc.bspyproc import get_processor


def perturb_data(configs, steps=1):
    # Load input data
    input_file = configs["data"]['input_data_file']
    output_file = configs['data']['perturbed_data_file']
    electrodes = configs['perturbation']['electrodes']
    perturb_fraction = configs['perturbation']['perturb_fraction']
    inputs, outputs, info = test_model.load_data(input_file, steps)
    # Perturb the data of the required electrodes
    for i in electrodes:
        amplitude = perturb_fraction * (inputs[:, i].max() - inputs[:, i].min())
        inputs[:, i] = inputs[:, i] + np.random.uniform(low=-amplitude / 2, high=+amplitude / 2, size=inputs[:, i].shape)
    # Save perturbed data such that it can be read by the (existing) test_model.get_error
    np.savez(output_file, inputs=inputs, outputs=outputs, info=info)
    return inputs, outputs, info

def get_perturbed_rmse(configs):
    # Load config data
    electrodes_sets = configs['perturbation']['electrodes_sets']
    perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']
    # Calculate the errors
    rmse = np.zeros((len(perturb_fraction_sets), len(electrodes_sets)))
    for i in range(len(perturb_fraction_sets)):
        configs['perturbation']['perturb_fraction'] = perturb_fraction_sets[i]
        for j in range(len(electrodes_sets)):
            configs['perturbation']['electrodes']= electrodes_sets[j]
            # Perturb data
            perturb_data(configs)
            # Get error
            rmse[i, j] = np.sqrt(test_model.get_error(configs['processor']["torch_model_dict"], configs["data"]['perturbed_data_file']))
    return rmse

if __name__ == "__main__":
    # User variables
    configs = load_configs('configs/validation/perturbation_configs.json')
    rmse = get_perturbed_rmse(configs)

    # Visualize results
    import matplotlib.pyplot as plt
    electrodes_sets = configs['perturbation']['electrodes_sets']
    perturb_fraction_sets = configs['perturbation']['perturb_fraction_sets']
    plt.figure()
    for j in range(len(electrodes_sets)):
        plt.plot(perturb_fraction_sets, rmse[:, j], marker='s', linestyle='')  # or use plt.semilogy(..)
    plt.xlabel('Perturbation fraction')
    plt.ylabel('RMSE (nA)')
    #legend_entries = (np.array(['Electrode']*len(electrodes_sets)).flatten().astype(str) + np.array(electrodes_sets).flatten().astype(str) ).tolist()
    plt.legend(electrodes_sets)
    plt.grid()

    # Fitting linear
    from sklearn import linear_model
    plt.gca().set_prop_cycle(None)  # reset color cycle, such that we have the same colors for the fitted lines as for the markers
    for j in range(len(electrodes_sets)):
        y = rmse[:, j]
        X = np.c_[perturb_fraction_sets]
        sample_weight = np.ones_like(y)
        sample_weight[0] = 1  # extra weigth for the first point, to fix the offset of baseline error
        clf = linear_model.LinearRegression(fit_intercept=True).fit(X, y, sample_weight)
        X_test = np.c_[np.arange(min(perturb_fraction_sets), max(perturb_fraction_sets), 0.001)]
        plt.plot(X_test, clf.predict(X_test))

    plt.title('RMSE scaling: simulated (square markers) and linear fit (solid line)')
