import numpy as np
import torch
from function import *
import matplotlib.pyplot as plt
import matplotlib as mpl

from metric_learning.process_dataset import vowel_dataset
from metric_learning.loss import calc_pair_acc


fig, axes = plt.subplots(nrows=2, ncols=3)

_, test_data = vowel_dataset(train_ratio=0.7)
x_test, y_test = torch.tensor(test_data[0]), torch.tensor(test_data[1])

norm = mpl.colors.TwoSlopeNorm(vmin=0.5, vcenter=0.9, vmax=1)

i = 0
for folder in ['metric_n=1_0.9', 'metric_n=2_0.9']:

    for step in [0, 90, 180]:

        probs_test = np.load(f'experimental_raw_data/{folder}/exp_data_test/step_{step:03d}_test.npy', allow_pickle=True).item()['exp']
        matrix = probs_test @ probs_test.T
        test_pair_acc = calc_pair_acc(probs_test, y_test, threshold=0.9)

        ax = axes.flat[i]
        ax.imshow(matrix, cmap='GnBu', norm=norm)
        ax.set_title(f'{test_pair_acc*100:.2f}%')
            
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.set_ylabel('n=1')
        elif i == 3:
            ax.set_ylabel('n=2')
        
        if i >= 3:        
            ax.set_xlabel(f'Epoch {step}')

        i += 1


cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap='GnBu', norm=norm), ax=axes.ravel().tolist())
cbar.ax.set_yticks([1, 0.8, 0.6])
cbar.ax.set_yticklabels(['1', '0.8', '0.6'])

plt.show()
