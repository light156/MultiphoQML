import torch
from torch import nn
import torch.nn.functional as F
from function import *


class BQNN_simulation(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.in_config = args['in_config']
        self.out_config_list = args['out_config_list']
        
        self.clements_modes = clements_mode_list(mode_num=6) 
        self.params = nn.Parameter(torch.randn(len(self.clements_modes)*2)) # 15*2
        self.output_phase = nn.Parameter(torch.randn(6))

        self.all_modes_list = [[1, 2], [3, 4], [5, 6], [2, 3], [4, 5]]*2

        self.param_phi = nn.Parameter(torch.randn(4))
        self.param_theta = nn.Parameter(torch.randn(4))

        self.input_k = nn.Parameter(torch.randn(12))
        self.input_b = nn.Parameter(torch.randn(12))


    def get_batch_ansatz(self, batch_x):
        batch_theta_phi_list = []

        for x in batch_x:
            x_scaled = x*self.input_k+self.input_b

            phi_list = list(x_scaled[0:3])
            theta_list = list(x_scaled[3:6])

            phi_list += list(self.param_phi[0:2])
            theta_list += list(self.param_theta[0:2])

            phi_list += list(x_scaled[6:9])
            theta_list += list(x_scaled[9:12])
            
            phi_list += list(self.param_phi[2:4])
            theta_list += list(self.param_theta[2:4])

            batch_theta_phi_list.append((theta_list, phi_list))

        return batch_theta_phi_list


    def calc_state(self, batch_x):
        
        start_unitary = calc_unitary(6, self.clements_modes, phi_list=self.params[0:15], theta_list=self.params[15:30])
        start_unitary = torch.diag(torch.exp(1j*self.output_phase)) @ start_unitary

        batch_theta_phi_list = self.get_batch_ansatz(batch_x)
        
        temp_list = []

        for theta_list, phi_list in batch_theta_phi_list:

            final_unitary = calc_unitary(6, self.all_modes_list, theta_list, phi_list) @ start_unitary
            # print(theta_list, phi_list, final_unitary)
            temp_list.append(calc_boson_sampling(final_unitary, self.in_config, self.out_config_list))

        complex_probs = torch.stack(temp_list)
        complex_probs = F.normalize(complex_probs, p=2)
        return complex_probs

    def forward(self, x):
        return torch.abs(self.calc_state(x))
    

class BQNN_hardware(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.in_config = args['in_config']
        self.out_config_list = args['out_config_list']
        self.photon_num = args['photon_num']
        
        self.all_modes_list = [[1, 2], [5, 6], [2, 3], [4, 5]] + [[1, 2], [3, 4], [5, 6], [2, 3], [4, 5]]*2

        self.param_phi = nn.Parameter(torch.zeros(4))
        self.param_theta = nn.Parameter(torch.zeros(8))

        self.input_k = nn.Parameter(torch.ones(12))
        self.input_b = nn.Parameter(torch.zeros(12))

    def get_batch_ansatz(self, batch_x):
        batch_theta_phi_list = []

        for x in batch_x:
            
            x_scaled = x*self.input_k+self.input_b

            theta_list, phi_list = list(self.param_theta[0:4]), list(torch.zeros(4))

            phi_list += list(x_scaled[0:3])
            theta_list += list(x_scaled[3:6])

            phi_list += list(self.param_phi[0:2])
            theta_list += list(self.param_theta[4:6])

            phi_list += list(x_scaled[6:9])
            theta_list += list(x_scaled[9:12])
            
            phi_list += list(self.param_phi[2:4])
            theta_list += list(self.param_theta[6:8])

            batch_theta_phi_list.append((theta_list, phi_list))

        return batch_theta_phi_list

    def calc_state(self, batch_x):
        batch_theta_phi_list = self.get_batch_ansatz(batch_x)
        
        temp_list = []

        for theta_list, phi_list in batch_theta_phi_list:

            final_unitary = calc_unitary(6, self.all_modes_list, theta_list, phi_list)
            # print(theta_list, phi_list, final_unitary)
            if self.photon_num == 2:
                temp_list.append(calc_boson_sampling(final_unitary, self.in_config, self.out_config_list))
            else:
                temp_list.append(final_unitary @ self.in_config)

        complex_probs = torch.stack(temp_list)
        complex_probs = F.normalize(complex_probs, p=2)
        return complex_probs

    def forward(self, x):
        return torch.abs(self.calc_state(x))
    