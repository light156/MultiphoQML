import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

from function import *
from metric_learning.process_dataset import vowel_dataset
from metric_learning.loss import *


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

_, test_data = vowel_dataset(train_ratio=0.7)
x_test, y_test = torch.tensor(test_data[0]), torch.tensor(test_data[1])

folder = 'experimental_raw_data/metric_n=2_0.95'
loss_fn = PairwiseLoss(threshold=0.95, margin=0)


N = 150
loss_train, loss_test = [], []

for step in range(1, N+1):
    exp_data_train = np.load(f'{folder}/exp_data_test/step_{step:03d}_train.npy', allow_pickle=True).item()
    loss_train.append(exp_data_train['loss'])


for step in range(0, N+1, 5):
    exp_data_test = np.load(f'{folder}/exp_data_test/step_{step:03d}_test.npy', allow_pickle=True).item()
    loss_test.append(loss_fn(exp_data_test['exp'], y_test).detach().numpy())


ax1.plot(range(1, N+1), loss_train[:N], label='Train')
ax1.plot(range(0, N+1, 5), loss_test[:(N//5)+1], label='Test')

ax1.set_xticks([0, 50, 100, 150])
ax1.set_xlim([-5, N+5])
ax1.set_yscale('log')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Exp loss')
ax1.legend()


exp_data_test = np.load(f'{folder}/exp_data_test/step_170_test.npy', allow_pickle=True).item()
exp_bs_probs = exp_data_test['exp']
exp_matrix = torch.matmul(exp_bs_probs, exp_bs_probs.T).detach().numpy()

norm = mpl.colors.TwoSlopeNorm(vmin=0.6, vcenter=0.95, vmax=1.0)

ax2.imshow(exp_matrix, norm=norm, cmap='coolwarm')
ax2.set_xticks([])
ax2.set_yticks([])

cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'), extend='min')

cbar.ax.set_yscale('log')
cbar.ax.set_yticks([1, 0.95, 0.9, 0.8, 0.7, 0.6])
cbar.ax.set_yticklabels(['1', '0.95', '0.9', '0.8', '0.7', '0.6'])

plt.show()
