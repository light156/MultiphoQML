import numpy as np
import matplotlib.pyplot as plt


y_values= [10, 18, 24, 28, 30]

plt.xlim([0, 25])
plt.ylim([0, 32])
plt.xticks(list(range(0, 26, 10)), labels=[f'{2*i}' for i in range(0, 26, 10)])
plt.yticks(y_values)

for i, photon_num in enumerate([1,2,3,4,5]):
    rank_list = np.load(f'fig2/qfim_data/qfim_mode_6_photon_{photon_num}.npy')
    p = plt.plot(range(26), rank_list, label=f'n = {photon_num}')
    plt.hlines(y_values[i], plt.xlim()[0], 60, linestyles='dashed', colors=p[0].get_color())

plt.xlabel('parameter numbers '+r'$M$')
plt.ylabel('QFIM rank')
plt.legend()
plt.show()
