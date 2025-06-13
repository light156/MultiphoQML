from utils import *
import numpy as np
from function import *

# the random seed for fixing the program result
seed = 0
np.random.seed(seed)

# port number and photon number
mode_num = 5
photon_num = 2 

# the unitary to learn, and the output basis used for training
target_unitary = np.load('experimental_raw_data/unitary_5m2p/target_unitary.npy') # replace with the correct folder

# the number of training states, which does not include the possible second single-photon round
n_training = 2

# preparing unique fock states for training
clements_mode = [[1,2],[3,4],[2,3],[4,5],[1,2],[3,4],[2,3],[4,5],[1,2],[3,4]]

params_min = np.load('experimental_raw_data/unitary_5m2p/params/params_1385.npy') # replace with the best epoch

current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                     phi_list=params_min[:len(clements_mode)], theta_list=params_min[len(clements_mode):])

swap_unitary = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,1,0]]) # for 2 photons case

first_round_unitary = swap_unitary @ current_unitary @ target_unitary

# define trainable parameters and SPSA settings
output_phase = np.random.randn(mode_num)

# chip setup and functions
output_photon_basis = [[i, j] for i in range(1, 6) for j in range(i+1, 7)]
six_port_chip = np.eye(6, dtype=np.complex128)

print("\nStart training phase by using single photon using trained circuit+SWAP")

training_phase_prep_unitary = random_unitary(mode_num)

c, gamma = 0.4, 0.101
a, alpha = 3, 0.602

for epoch in range(1, 201):

    c_k = c / (epoch**gamma)
    a_k = a / (epoch**alpha)

    random_binary = np.random.randint(0, 2, mode_num)*2-1
    output_phase_pos, output_phase_neg = output_phase+c_k*random_binary, output_phase-c_k*random_binary
    
    ##################### pos #####################
    output_phase_unitary = np.diag(np.exp(1j*output_phase_pos))
    temp_unitary = training_phase_prep_unitary.T.conj() @ output_phase_unitary @ first_round_unitary @ training_phase_prep_unitary

    six_port_chip[1:, 1:] = temp_unitary
    
    bs_theo = calc_bs_two_photons(six_port_chip, [1,6], output_photon_basis, identical_photon=False)
    bs_exp = bs_theo/np.sum(bs_theo)

    train_loss_pos = bs_exp[0]+bs_exp[1]+bs_exp[2]+bs_exp[3]

    ##################### neg #####################
    output_phase_unitary = np.diag(np.exp(1j*output_phase_neg))
    temp_unitary = training_phase_prep_unitary.T.conj() @ output_phase_unitary @ first_round_unitary @ training_phase_prep_unitary

    six_port_chip[1:, 1:] = temp_unitary
    
    bs_theo = calc_bs_two_photons(six_port_chip, [1,6], output_photon_basis, identical_photon=False)
    bs_exp = bs_theo/np.sum(bs_theo)
    fid = np.sum(np.sqrt(bs_exp*bs_theo))

    train_loss_neg = bs_exp[0]+bs_exp[1]+bs_exp[2]+bs_exp[3]

    ##################### update gradients #####################
    const = (train_loss_pos-train_loss_neg)/2/c_k

    output_phase_grad = const/random_binary
    output_phase -= a_k*output_phase_grad

    ##################### real loss #####################
    end = time.time()

    output_phase_unitary = np.diag(np.exp(1j*output_phase))
    temp_unitary = training_phase_prep_unitary.T.conj() @ output_phase_unitary @ first_round_unitary @ training_phase_prep_unitary
    
    train_loss = np.sum(np.abs(temp_unitary[:4, 4])**2)

    if epoch%10 == 0:
        print(f'Epoch {epoch}, Train loss: {train_loss:>.6f}, Out probs: {np.abs(temp_unitary[4, 4])**2:>.6f}')

    if train_loss<1e-5:
        break


final_unitary_phase = np.diag(np.exp(1j*output_phase))
final_unitary_full = final_unitary_phase @ first_round_unitary

######################## final testing after training #####################

n_test = 10
test_prep_unitary=[]

for k in range(n_test):
    test_prep_unitary.append(random_unitary(mode_num))

error_test_list=[]

for k in range(n_test):
    output_probs = np.abs(test_prep_unitary[k].T.conj() @ final_unitary_full @ test_prep_unitary[k])[:, 0]**2
    error_t = np.sum(output_probs[1:])
    error_test_list.append(error_t)
        
error_test=np.mean(error_test_list)

print("\nFinished training, test error", error_test)
