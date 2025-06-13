# Plot the training loss and validation loss in real physical experiments

import numpy as np
import torch
import matplotlib.pyplot as plt

from function import *
from metric_learning.process_dataset import vowel_dataset
from metric_learning.loss import *


_, test_data = vowel_dataset(train_ratio=0.7)
x_test, y_test = torch.tensor(test_data[0]), torch.tensor(test_data[1])

loss_fn = PairwiseLoss(threshold=0.9, margin=0)
N = 200

plt.figure(figsize=(8,4))

################## threshold 0.9, 1 photon ##################

folder = 'experimental_raw_data/metric_n=1_0.9'
loss_train, loss_test = [], []


for step in range(1, N+1):
    exp_data_train = np.load(f'{folder}/exp_data_test/step_{step:03d}_train.npy', allow_pickle=True).item()
    loss_train.append(exp_data_train['loss'])


for step in range(0, N+1, 5):
    exp_data_test = np.load(f'{folder}/exp_data_test/step_{step:03d}_test.npy', allow_pickle=True).item()
    loss_test.append(loss_fn(exp_data_test['exp'], y_test).detach().numpy())


plt.plot(range(1, N+1), loss_train[:N], label='n=1 training', color='dodgerblue', linestyle='dashed')
plt.plot(range(0, N+1, 5), loss_test[:(N//5)+1], label='n=1 test', color='dodgerblue')


################## threshold 0.9, 2 photon ##################

folder = 'experimental_raw_data/metric_n=2_0.9'
loss_train, loss_test = [], []


for step in range(1, N+1):
    exp_data_train = np.load(f'{folder}/exp_data_test/step_{step:03d}_train.npy', allow_pickle=True).item()
    loss_train.append(exp_data_train['loss'])


for step in range(0, N+1, 5):
    exp_data_test = np.load(f'{folder}/exp_data_test/step_{step:03d}_test.npy', allow_pickle=True).item()
    loss_test.append(loss_fn(exp_data_test['exp'], y_test).detach().numpy())


plt.plot(range(1, N+1), loss_train[:N], label='n=2 training', color='red', linestyle='dashed')
plt.plot(range(0, N+1, 5), loss_test[:(N//5)+1], label='n=2 test', color='red')


plt.xticks([0, 60, 120, 180])
plt.xlim([-10, N])
plt.xlabel('Epoch')
plt.ylabel('Exp loss')
plt.yscale('log')

plt.legend()
plt.show()
