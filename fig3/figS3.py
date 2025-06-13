import numpy as np
import matplotlib.pyplot as plt
from function import *


mode_num = 5
clements_mode = [[1,2],[3,4],[2,3],[4,5],[1,2],[3,4],[2,3],[4,5],[1,2],[3,4]]

target_unitary = np.load(f'experimental_raw_data/unitary_L=2_n=2/target_unitary.npy')
target_unitary_other = np.load(f'experimental_raw_data/unitary_L=4_n=1/target_unitary.npy')
assert np.array_equal(target_unitary, target_unitary_other)

################## n=2 ##################
params = np.load(f'experimental_raw_data/unitary_L=2_n=2/params_1385.npy')
current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                    phi_list=params[:len(clements_mode)], theta_list=params[len(clements_mode):])

first_round_unitary = current_unitary @ target_unitary

train_loss_list, distance_list = [], []

for epoch in range(0, 201):
    output_phase = np.load(f'experimental_raw_data/unitary_L=2_n=2_second_round/output_phase_{epoch:04d}.npy')
    phase_unitary = np.diag(np.exp(1j*output_phase))

    final_unitary = phase_unitary @ first_round_unitary
    distance = np.abs(np.trace(final_unitary))/5
    distance_list.append(1-distance)

plt.plot(distance_list, label='L=2, n=2', color='blue', linewidth=1.5)

################## n=1 ##################
params = np.load(f'experimental_raw_data/unitary_L=4_n=1/params_1486.npy')
current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                    phi_list=params[:len(clements_mode)], theta_list=params[len(clements_mode):])

first_round_unitary = current_unitary @ target_unitary

distance_list = []

for epoch in range(0, 201):
    output_phase = np.load(f'experimental_raw_data/unitary_L=4_n=1_second_round/output_phase_{epoch:04d}.npy')
    phase_unitary = np.diag(np.exp(1j*output_phase))

    final_unitary = phase_unitary @ first_round_unitary
    distance = np.abs(np.trace(final_unitary))/5
    distance_list.append(1-distance)

plt.plot(distance_list, label='L=4, n=1', color='red', linewidth=1.5)

plt.plot(range(181), [0]*181, linestyle='dashed', color='black')

plt.xticks([0,50,100,150])
plt.xlim([-5, 180])
plt.xlabel('Epoch')
plt.ylabel('Matrix closeness')
plt.legend()
plt.show()