import numpy as np
import matplotlib.pyplot as plt


y_values= [18,31,35]

plt.xlim([0, 40])
plt.xticks(list(range(0, 41, 10)), labels=[f'{2*i}' for i in range(0, 41, 10)])
plt.yticks(y_values)

for i, data_num in enumerate([1,2,3]):
    rank_list = np.load(f'fig2/qfim_data/dqfim_mode_6_photon_2_data_{data_num}.npy')
    p = plt.plot(range(41), rank_list, label=f'L = {data_num}')
    plt.hlines(y_values[i], plt.xlim()[0], 60, linestyles='dashed', colors=p[0].get_color())

plt.xlabel('parameter numbers '+r'$M$')
plt.ylabel('DQFIM rank')
plt.legend()
plt.show()
