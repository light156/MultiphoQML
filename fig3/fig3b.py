import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from function import *


mode_num = 5
clements_mode = [[1,2],[3,4],[2,3],[4,5],[1,2],[3,4],[2,3],[4,5],[1,2],[3,4]]

N = 1500

folder_list = ['L=2_n=2', 'L=3_n=1', 'L=4_n=1']
label_list = ['L=2, n=2', 'L=3, n=1', 'L=4, n=1']
color_list = ['blue', 'orange', 'red']

target_unitary_basis = np.load(f'experimental_raw_data/unitary_{folder_list[0]}/target_unitary.npy')


for i, folder in enumerate(folder_list):
    print(folder)
    target_unitary = np.load(f'experimental_raw_data/unitary_{folder}/target_unitary.npy')
    assert np.array_equal(target_unitary_basis, target_unitary)

    distance_list = []

    for epoch in range(0, N+1):
        params = np.load(f'experimental_raw_data/unitary_{folder}/params_{epoch:04d}.npy')
        current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                            phi_list=params[:len(clements_mode)], theta_list=params[len(clements_mode):])
        
        final_unitary = current_unitary @ target_unitary
        distance = 1-np.trace(np.abs(final_unitary))/mode_num
        distance_list.append(distance)
    
    print(np.argmin(distance_list))
    plt.plot(range(N+1), distance_list, label=label_list[i], color=color_list[i], linewidth=1.5)


plt.plot(range(1251), [0]*1251, linestyle='dashed', color='black', linewidth=1.5)

plt.xticks([0,400,800,1200])
plt.yticks([0,0.2,0.4,0.6])
plt.xlim([-20, 1250])

plt.xlabel('Epoch')
plt.ylabel('Matrix closeness')
plt.legend()
plt.show()
