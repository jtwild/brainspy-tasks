import numpy as np


def perceptron(wvfrm, target, tolerance=0.01, max_iter=200):
    # Assumes that the waveform wvfrm and the target have the shape (n_total,1)
    # Normalize the data; it is assumed that the target has binary values
    wvfrm = (wvfrm - np.mean(wvfrm)) / np.std(wvfrm)
    n_total = len(wvfrm)
    weights = np.random.randn(2, 1)  # np.zeros((2,1))
    inp = np.concatenate([np.ones_like(wvfrm), wvfrm], axis=1)
    shuffle = np.random.permutation(len(inp))

    n_test = int(0.25 * n_total)
    x_test = inp[shuffle[:n_test]]
    y_test = target[shuffle[:n_test]]

    x = inp[shuffle[n_test:]]
    y = target[shuffle[n_test:]]

    def f(x):
        return float(x < 0)

    error = np.inf
    j = 0
    while (error > tolerance) and (j < max_iter):

        for i in range(len(x)):
            a = np.dot(weights.T, x[i])
            delta = y[i] - f(a)
            weights = weights - delta * x[i][:, np.newaxis]

        predict = np.array(list(map(f, np.dot(x_test, weights))))
        predict = predict[:, np.newaxis]
        error = np.mean(np.abs(y_test - predict))
        j += 1
#        print('Prediction Error: ',error, ' in ', j,' iters')

    buffer = np.zeros_like(y_test)
    buffer[y_test == predict] = 1
    n_correct = np.sum(buffer)
    accuracy = n_correct / n_test

    predicted = (shuffle[:n_test], predict)
#    print('Fraction of iterations used: ', j/max_iter)
#    pdb.set_trace()
    corrcoef = np.corrcoef(y_test.T, x_test[:, 1].T)[0, 1]
    if accuracy > 0.9 and weights[0] < 0 and corrcoef < 0:
        print('Weight is negative', weights[0], ' and correlation also: ', corrcoef)
        accuracy = 0.
        print('Accuracy is set to zero!')

    return accuracy, weights, predicted
