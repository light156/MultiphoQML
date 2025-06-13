from function import *
from math import pi
from functools import partial
import multiprocess as mp
import os


if __name__ == '__main__':


    def calc_dqfim_rank(seed, mode_num, photon_num, mzi_num, data_num):

        random.seed(seed)    
        torch.manual_seed(seed)

        modes_list = random_modes_list(mode_num, mzi_num)

        in_config = [1]*photon_num+[0]*(mode_num-photon_num)
        output_config_list = get_out_configs(mode_num, photon_num, if_bunching=True)

        param_theta = torch.rand(mzi_num, dtype=torch.float64, requires_grad=True) * pi * 2
        param_phi = torch.rand(mzi_num, dtype=torch.float64, requires_grad=True) * pi * 2
        trainable_unitary = calc_unitary(mode_num, modes_list, param_theta, param_phi)

        temp_sum = 0

        for _ in range(data_num):
            prep_unitary = torch.as_tensor(random_unitary(mode_num))
            state = calc_boson_sampling(trainable_unitary@prep_unitary, in_config, output_config_list)
            state_detached = state.clone().detach()
            temp_sum += torch.vdot(state_detached, state)

        fs_dist = 2 - 2.0 * torch.abs(temp_sum/data_num)**2
        hes_matrix = torch.real(hessian(fs_dist, (param_theta, param_phi)))
        eig, _ = np.linalg.eig(np.asarray(hes_matrix))
        return eig
        

    ############## Calculate the rank #############
    mode_num = 6
    photon_num = 2
    mzi_total_num = 40
    circuit_num = 3
    data_num_list = [1,2,3]


    data_folder = 'fig2/qfim_data'

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for data_num in data_num_list:
        rank_list = [0]

        for mzi_num in range(1, mzi_total_num+1, 1):
            print('data num', data_num, 'param num', mzi_num*2)
            
            with mp.Pool(circuit_num) as pool:
                temp_func = partial(calc_dqfim_rank, mode_num=mode_num, photon_num=photon_num, mzi_num=mzi_num, data_num=data_num)
                eig = pool.map(temp_func, range(circuit_num))

            eig_num = np.mean(np.sum(np.array(eig)>1e-10, axis=1))
            rank_list.append(eig_num)
        
        np.save(os.path.join(data_folder, f'dqfim_mode_{mode_num}_photon_{photon_num}_data_{data_num}.npy'), rank_list)
