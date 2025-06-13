import numpy as np
import torch
from function import *
import matplotlib as mpl
import matplotlib.pyplot as plt


exp_data_test = np.load(f'experimental_raw_data/metric_n=2_0.9/exp_data_test/step_180_test.npy', allow_pickle=True).item()
exp_bs_probs = exp_data_test['exp']
exp_matrix = torch.matmul(exp_bs_probs, exp_bs_probs.T).detach().numpy()

norm = mpl.colors.TwoSlopeNorm(vmin=0.5, vcenter=0.9, vmax=1.0)

plt.imshow(exp_matrix, norm=norm, cmap='coolwarm')
plt.xticks([])
plt.yticks([])

cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='coolwarm'), extend='min')

cbar.ax.set_yscale('log')
cbar.ax.set_yticks([1, 0.9, 0.8, 0.7, 0.6, 0.5])
cbar.ax.set_yticklabels(['1', '0.9', '0.8', '0.7', '0.6', '0.5'])

plt.show()
