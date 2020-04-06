
from bspyalgo.utils.io import load_configs
from bspysmg.model.data.outputs import test_model
from bspysmg.model.data.outputs.train_model import train_surrogate_model
import numpy as np
from bspyproc.bspyproc import get_processor


def perturb_data(input_file, output_file, electrodes, perturb_fraction=0.3, steps=1):
    inputs, outputs, info = test_model.load_data(input_file, steps)
    for i in electrodes:
        amplitude = perturb_fraction * (inputs[:, i].max() - inputs[:, i].min())
        inputs[:, i] = inputs[:, i] + np.random.uniform(low=-amplitude / 2, high=+amplitude / 2, size=inputs[:, i].shape)
    np.savez(output_file, inputs=inputs, outputs=outputs, info=info)
    return inputs, outputs, info


if __name__ == "__main__":
    # User variables
    configs = load_configs('configs/validation/perturbation_configs.json')
    electrodes_sets = [[0], [1], [3], [4]]
    perturb_fraction_sets = [0.1, 0.2, 0.3, 0.4]

    # Calculate the errors
    mse = np.zeros((len(perturb_fraction_sets), len(electrodes_sets)))
    for i in range(len(perturb_fraction_sets)):
        perturb_fraction = perturb_fraction_sets[i]
        for j in range(len(electrodes_sets)):
            electrodes = electrodes_sets[j]
            # Perturb data
            perturb_data(configs["data"]['test_data_path'], configs['data']['perturbed_data_path'], electrodes, perturb_fraction=perturb_fraction)
            # Get error
            mse[i, j] = test_model.get_error(configs['processor']["torch_model_dict"], configs["data"]['perturbed_data_path'])

    # Visualize results
    import matplotlib.pyplot as plt
    plt.figure()
    for j in range(len(electrodes_sets)):
        plt.plot(perturb_fraction_sets, mse[:, j], marker='s')  # or use plt.semilogy(..)
    plt.xlabel('Perturbation fraction')
    plt.ylabel('MSE (nA)')
    plt.legend(electrodes_sets)
    plt.grid()

    # Fitting quadratic
    j=0
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    degree=1
    quadr=PolynomialFeatures(degree=degree, include_bias=False)
    y = mse[:,j]
    X = np.matmul(np.c_[perturb_fraction_sets], np.ones((1,degree)))
    X_=quadr.fit_transform(X)

    clf = linear_model.LinearRegression()
    clf.fit(X_, y)
    X_test = np.matmul(np.arange(0,1,0.01).reshape(-1,1), np.ones((1,degree)))
    X_test_ = quadr.fit_transform(X_test)
    plt.plot(X_test, clf.predict(X_test_))
