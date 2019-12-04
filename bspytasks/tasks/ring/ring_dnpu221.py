'''Experiment to test classification of ring data with a 2-2-1 DNPU architecture. Uses the negative Fischer function as cost function. 
authout: H. C. Ruiz
'''

import numpy as np
import matplotlib.pyplot as plt
from dnpu_221 import DNPU_NET
from bspyproc.processors.simulation.dopanet import DNPU
from sgd_torch import trainer
from bspyproc.utils.pytorch import TorchUtils
import torch
# Configure the dnpu network
IN_DICT = {}
IN_DICT['input_node1'] = [3, 4]
IN_DICT['input_node2'] = [3, 4]
IN_DICT['hidden_node1'] = [3, 4]
IN_DICT['hidden_node2'] = [3, 4]
IN_DICT['output_node'] = [3, 4]

# Initialize Data
with np.load(f'../Data/Inputs/Ring_data/Class_data_0.0125.npz') as data:
    TARGETS = data['target']
    TARGETS = 1 - TARGETS[:, np.newaxis]
    INPUTS = data['inp_wvfrm']

# plt.figure()
# plt.plot(INPUTS[:, 0], INPUTS[:, 1], 'o')
# plt.show()

TARGETS = TorchUtils.get_tensor_from_numpy(TARGETS)
INPUTS = TorchUtils.get_tensor_from_numpy(INPUTS)
training_data = (INPUTS, TARGETS)


def fisher(x, y):
    '''Separates classes irrespective of assignments.
    Reliable, but insensitive to actual classes'''
    x_high = x[(y == 1)]
    x_low = x[(y == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1-m0)**2
    return -mean_separation / (s0 + s1)


RUNS = 1000
outputs_dnpu221 = np.zeros((RUNS,)+TARGETS.shape)
outputs_dnpu = np.zeros((RUNS,)+TARGETS.shape)
final_costs_dnpu221 = np.zeros(RUNS)
final_costs_dnpu = np.zeros(RUNS)
corr_dnpu221 = np.zeros(RUNS)
corr_dnpu = np.zeros(RUNS)
# =============== Training ============= #
nr_eps = 1200
lr = 0.001  # lr = 1e-3 works
for run in range(RUNS):
    print(f'{"#"*20} RUN {run+1} {"#"*20}')
    # --------------- DNPU 221 --------------- #
    dnpu_221 = DNPU_NET(IN_DICT)
    costs = trainer(dnpu_221, training_data, loss_fn=fisher,
                    learning_rate=lr, nr_epochs=nr_eps, batch_size=110)

    prediction = dnpu_221(INPUTS).cpu().detach().numpy()
    correlation = np.corrcoef(
        prediction.T, TARGETS.cpu().numpy().T)[0, 1]
    print(f'Target correlation for dnpu_221: {correlation}')
    outputs_dnpu221[run] = prediction
    final_costs_dnpu221[run] = -costs[-1, 0]
    corr_dnpu221[run] = correlation
    # --------------- DNPU --------------- #
    dnpu = DNPU(IN_DICT['input_node1'],
                path=r'../Data/Models/checkpoint3000_02-07-23h47m.pt')
    costs = trainer(dnpu, training_data, loss_fn=fisher,
                    learning_rate=lr, nr_epochs=nr_eps, batch_size=110)

    prediction = dnpu(INPUTS).cpu().detach().numpy()
    correlation = np.corrcoef(
        prediction.T, TARGETS.cpu().numpy().T)[0, 1]
    print(f'Target correlation for dnpu: {correlation}')
    outputs_dnpu[run] = prediction
    final_costs_dnpu[run] = -costs[-1, 0]
    corr_dnpu[run] = correlation

plt.figure()
plt.hist([final_costs_dnpu, final_costs_dnpu221], 30)
plt.show()

plt.figure()
plt.plot(final_costs_dnpu, corr_dnpu, 'ob')
plt.plot(final_costs_dnpu221, corr_dnpu221, 'xr')
plt.show()

np.savez('../Data/Results/comparison_single-vs-221.npz',
         final_costs_dnpu=final_costs_dnpu,
         corr_dnpu=corr_dnpu,
         outputs_dnpu=outputs_dnpu,
         final_costs_dnpu221=final_costs_dnpu221,
         corr_dnpu221=corr_dnpu221,
         outputs_dnpu221=outputs_dnpu221)

plt.figure()
plt.plot(outputs_dnpu[np.nanargmax(final_costs_dnpu)])
plt.plot(outputs_dnpu221[np.nanargmax(final_costs_dnpu221)])
plt.show()

print('DONE')
