from utils import *
from function import *
import numpy as np


# the random seed for fixing the program result
seed = 0
np.random.seed(seed)

# port number and photon number
mode_num = 5
photon_num = 1 

# the unitary to learn, and the output basis used for training
target_unitary = random_unitary(mode_num)

# the number of training states, which does not include the possible second single-photon round
n_training = 4
n_test = 10

# preparing unique fock states for training
clements_mode = [[1,2],[3,4],[2,3],[4,5],[1,2],[3,4],[2,3],[4,5],[1,2],[3,4]]
hamming_dist_train = np.array([[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]])

# define trainable parameters and SPSA settings
params = np.random.randn(len(clements_mode)*2)
output_phase = np.random.randn(mode_num)

################# Experiment Start ######################
slice_from_6_to_5 = [0, 1, 2, 3, 4]
output_photon_basis = [[i, j] for i in range(1, 6) for j in range(i+1, 7)]
six_port_chip = np.eye(6, dtype=np.complex128)


################# Experiment Start ######################
def set_port_and_measure(exp_in_config, unitary):

    ########## replace experiment code ################
    # set_all_phase(chip_phase_array)
    # time.sleep(2) # chip setting time
    # data = measure(server, step=1)['cc'][0]
    # out_probs = data/np.sum(data)
    
    data = calc_bs_two_photons(unitary, exp_in_config, output_photon_basis, False)
    out_probs = data/np.sum(data)

    return out_probs, None, None


################  First round, where unitaries can be learned under swap operations ################  
print("Start training")

c, gamma = 0.4, 0.101
a, alpha = 3, 0.602

train_loss_min, epoch_min, params_min = 100, 0, None

start_epoch = 0

for epoch in range(start_epoch+1, 2001):

    c_k = c / (epoch**gamma)
    a_k = a / (epoch**alpha)
    
    random_binary = np.random.randint(0, 2, len(params))*2-1
    params_pos, params_neg = params+c_k*random_binary, params-c_k*random_binary

    ##################### pos #####################
    current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                         phi_list=params_pos[:len(clements_mode)], theta_list=params_pos[len(clements_mode):])
    
    six_port_chip[1:, 1:] = current_unitary @ target_unitary

    # set_uni(six_port_chip)
    # time.sleep(2) # chip setting time

    exp_probs_13_pos, _, _ = set_port_and_measure((1,3), six_port_chip)
    train_loss_pos = np.sum(hamming_dist_train[0]*exp_probs_13_pos[slice_from_6_to_5])

    exp_probs_14_pos, _, _ = set_port_and_measure((1,4), six_port_chip)
    train_loss_pos += np.sum(hamming_dist_train[1]*exp_probs_14_pos[slice_from_6_to_5])

    exp_probs_15_pos, _, _ = set_port_and_measure((1,5), six_port_chip)
    train_loss_pos += np.sum(hamming_dist_train[2]*exp_probs_15_pos[slice_from_6_to_5])

    exp_probs_16_pos, _, _ = set_port_and_measure((1,6), six_port_chip)
    train_loss_pos += np.sum(hamming_dist_train[3]*exp_probs_16_pos[slice_from_6_to_5])

    ##################### neg #####################
    current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                         phi_list=params_neg[:len(clements_mode)], theta_list=params_neg[len(clements_mode):])
    
    six_port_chip[1:, 1:] = current_unitary @ target_unitary

    # set_uni(six_port_chip)
    # time.sleep(2) # chip setting time

    exp_probs_13_neg, _, _ = set_port_and_measure((1,3), six_port_chip)
    train_loss_neg = np.sum(hamming_dist_train[0]*exp_probs_13_neg[slice_from_6_to_5])

    exp_probs_14_neg, _, _ = set_port_and_measure((1,4), six_port_chip)
    train_loss_neg += np.sum(hamming_dist_train[1]*exp_probs_14_neg[slice_from_6_to_5])

    exp_probs_15_neg, _, _ = set_port_and_measure((1,5), six_port_chip)
    train_loss_neg += np.sum(hamming_dist_train[2]*exp_probs_15_neg[slice_from_6_to_5])

    exp_probs_16_neg, _, _ = set_port_and_measure((1,6), six_port_chip)
    train_loss_neg += np.sum(hamming_dist_train[3]*exp_probs_16_neg[slice_from_6_to_5])

    ##################### update gradients #####################
    const = (train_loss_pos-train_loss_neg)/2/c_k

    params_grad = const/random_binary
    params -= a_k*params_grad

    ##################### real loss #####################
    current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                         phi_list=params[:len(clements_mode)], theta_list=params[len(clements_mode):])
    
    six_port_chip[1:, 1:] = current_unitary @ target_unitary

    probs_13 = calc_bs_two_photons(six_port_chip, (1,3), output_photon_basis, False)
    train_loss = np.sum(hamming_dist_train[0]*probs_13[slice_from_6_to_5])

    probs_14 = calc_bs_two_photons(six_port_chip, (1,4), output_photon_basis, False)
    train_loss += np.sum(hamming_dist_train[1]*probs_14[slice_from_6_to_5])
    
    probs_15 = calc_bs_two_photons(six_port_chip, (1,5), output_photon_basis, False)
    train_loss += np.sum(hamming_dist_train[2]*probs_15[slice_from_6_to_5])

    probs_16 = calc_bs_two_photons(six_port_chip, (1,6), output_photon_basis, False)
    train_loss += np.sum(hamming_dist_train[3]*probs_16[slice_from_6_to_5])

    if epoch%100==0:
        print(epoch, train_loss, probs_13[1], probs_14[2], probs_15[3], probs_16[4], a_k, c_k)
        # np.save(f'{folder}/params_{epoch:04d}.npy', params)
    
    if train_loss<train_loss_min:
        train_loss_min, epoch_min, params_min = train_loss, epoch, params


current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                     phi_list=params_min[:len(clements_mode)], theta_list=params_min[len(clements_mode):])

first_round_unitary = current_unitary @ target_unitary

print(epoch_min, train_loss_min)
