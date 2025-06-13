from utils import *
from function import *
import numpy as np


# the random seed for fixing the program result
seed = 0
np.random.seed(seed)

# port number and photon number
mode_num = 5
photon_num = 2 

# the unitary to learn, and the output basis used for training
target_unitary = random_unitary(mode_num)

# the number of training states, which does not include the possible second single-photon round
n_training = 2

# preparing unique fock states for training
clements_mode = [[1,2],[3,4],[2,3],[4,5],[1,2],[3,4],[2,3],[4,5],[1,2],[3,4]]

weight = 1
hamming_dist_train = np.array([[1,1,1,1,0,1,1,1,1,1]+[weight]*5,
                               [1,1,1,1,1,1,1,1,1,0]+[weight]*5])

# define trainable parameters and SPSA settings
params = np.random.randn(len(clements_mode)*2)
output_phase = np.random.randn(mode_num)

# chip setup and functions
output_photon_basis = [[i, j] for i in range(1, 5) for j in range(i+1, 6)]+[[i, i] for i in range(1, 6)]
six_port_chip = np.eye(6, dtype=np.complex128)
identical_photon = True

"""
# save configs
folder = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))

if not os.path.exists(folder):
    os.makedirs(folder)

np.save(f'{folder}/target_unitary.npy', target_unitary)
np.save(f'{folder}/params_0000.npy', params)
os.makedirs(f'{folder}/exp_data')
"""

####  First round, where unitaries can be learned under swap operations ####  

print("Start training")

c, gamma = 0.4, 0.101
a, alpha = 3, 0.602

train_loss_min, epoch_min, params_min = 100, 0, None

start_epoch = 0
# start_folder = folder
# params = np.load(f'{start_folder}/params_{start_epoch:04d}.npy')

for epoch in range(1, start_epoch+1):
    random_binary = np.random.randint(0, 2, len(params))*2-1


################# Experiment Start ######################
def set_port_and_measure(exp_in_config, unitary):

    ########## replace experiment code ################
    # set_all_phase(chip_phase_array)
    # time.sleep(2) # chip setting time
    # data = measure(server, step=1)['cc'][0]
    # out_probs = data/np.sum(data)
    
    data = calc_bs_two_photons(unitary, exp_in_config, output_photon_basis, identical_photon=identical_photon)
    out_probs = data/np.sum(data)

    return out_probs


for epoch in range(start_epoch+1, 2000):

    c_k = c / (epoch**gamma)
    a_k = a / (epoch**alpha)
    
    random_binary = np.random.randint(0, 2, len(params))*2-1
    params_pos, params_neg = params+c_k*random_binary, params-c_k*random_binary

    train_loss_pos, train_loss_neg = 0, 0
    
    ##################### pos #####################
    current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                         phi_list=params_pos[:len(clements_mode)], theta_list=params_pos[len(clements_mode):])

    # set_uni(six_port_chip)
    # time.sleep(2) # chip setting time

    bs_exp_results_pos = [set_port_and_measure(in_port, current_unitary @ target_unitary) for in_port in [(2,3),(4,5)]]
    for i in range(2):
        train_loss_pos += np.sum(hamming_dist_train[i]*bs_exp_results_pos[i])

    ##################### neg #####################
    current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                         phi_list=params_neg[:len(clements_mode)], theta_list=params_neg[len(clements_mode):])
    
    six_port_chip[1:, 1:] = current_unitary @ target_unitary

    # set_uni(six_port_chip)
    # time.sleep(2) # chip setting time
    
    bs_exp_results_neg = [set_port_and_measure(in_port, current_unitary @ target_unitary) for in_port in [(2,3),(4,5)]]
    for i in range(2):
        train_loss_neg += np.sum(hamming_dist_train[i]*bs_exp_results_neg[i])

    ##################### update gradients #####################
    const = (train_loss_pos-train_loss_neg)/2/c_k

    params_grad = const/random_binary
    params -= a_k*params_grad

    ##################### real loss #####################

    current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                         phi_list=params[:len(clements_mode)], theta_list=params[len(clements_mode):])
    
    U_theo = current_unitary @ target_unitary

    bs_theo_23 = calc_bs_two_photons(U_theo, (2,3), output_photon_basis, True)
    bs_theo_45 = calc_bs_two_photons(U_theo, (4,5), output_photon_basis, True)
    
    bs_theo_23 /= np.sum(bs_theo_23)
    bs_theo_45 /= np.sum(bs_theo_45)

    train_loss = np.sum(hamming_dist_train[0]*bs_theo_23)
    train_loss += np.sum(hamming_dist_train[1]*bs_theo_45)

    if epoch%100==0:
        print(f"Epoch {epoch}, Train loss: {train_loss:>.6f}, Out probs: 23: {bs_theo_23[4]:>.4f}, 45: {bs_theo_45[9]:>.4f}")
        # np.save(f'{folder}/params_{epoch:04d}.npy', params)
    
    if train_loss<train_loss_min:
        train_loss_min, epoch_min, params_min = train_loss, epoch, params


current_unitary = calc_unitary_numpy(mode_num, clements_mode, 
                                     phi_list=params_min[:len(clements_mode)], theta_list=params_min[len(clements_mode):])

first_round_unitary = current_unitary @ target_unitary

print(epoch_min, train_loss_min)

################  Second round, train local phases with a single photon ##################

print("\nStart training phase by using single photon using trained circuit+SWAP")

training_phase_prep_unitary = random_unitary(mode_num)

# prepared state before applying operations
probs_before = np.abs(training_phase_prep_unitary[:, 0])**2

# the output state after applying target unitary and trained unitary in the first round, which may have swapped modes
probs_after = np.abs(first_round_unitary @ training_phase_prep_unitary)[:, 0]**2

print(np.sort(probs_before))
print(np.sort(probs_after))

if not np.allclose(np.sort(probs_before), np.sort(probs_after), atol=5e-2):
    print("Cannot find permutation, not enough training states for learning unitary in the first step")
    exit()

##swap different modes such that probabilties matches
swap_graph = np.zeros(mode_num,dtype=int)

for k in range(len(probs_before)):
    swap_graph[k] = np.argmin(np.abs(probs_before-probs_after[k]))

assert np.all(np.sort(swap_graph)==np.arange(mode_num))

print("Found permutation", swap_graph)