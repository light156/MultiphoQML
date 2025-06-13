"""
Program to learn unitaries with linear photonics

Uses Fock states to reduce number of training states
One can train with Fock states, requiring m/n Fock training states (m modes, n photons) (+1 single photon state)
However, Fock states have permutation symmetries as they are indistinguishable. Also, one cannot learn single-mode phases with them
To fix this, one needs 2 additional steps

For Fock states, training involves in total three steps:
    1. Learn with multi-photon Fock state 
    2. Use a random 1-photon state to figure out swap permutation
    3. Learn the last layer of phases of circuit with the same 1-photon state in Step 2

# Originally written by Tobias Haug @TII
    tobias.haug@u.nus.edu
    
"""

import numpy as np
import torch
from torch import optim
from function import *

###################### basic setup ############################

# random seed for fixing the program result
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# port number and photon number
mode_num = 5
photon_num = 1
# the unitary to learn, and the output basis used for training
target_unitary = random_unitary(mode_num)
output_photon_basis = get_out_configs(mode_num, photon_num, if_bunching=True)

# the number of training states, which does not include the possible second 1-photon round
n_training = 4

# preparing unique fock states for training
fock = np.ones(mode_num,dtype=int)*(photon_num//mode_num)
fock[:(photon_num%mode_num)] += 1
training_state_list = [list(np.roll(fock,i*photon_num)) for i in range(n_training)]
print(training_state_list)

hamming_dist_train = []

for k in range(n_training):
    hamming_dist=np.zeros(len(output_photon_basis),dtype=float)

    ## distance of each photon state relative to initial state
    for i in range(len(output_photon_basis)):
        hamming_dist[i]=np.sum(np.abs(np.array(output_photon_basis[i])-np.array(training_state_list[k])))
    
    hamming_dist=hamming_dist/photon_num
    hamming_dist_train.append(torch.as_tensor(hamming_dist))


clements_mode = clements_mode_list(mode_num)

params = torch.randn(len(clements_mode)*2, dtype=torch.float64, requires_grad=True)
output_phase = torch.randn(mode_num, dtype=torch.float64, requires_grad=True)

optimizer = optim.Adam([params, output_phase], lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30)
################  First round, where unitaries can be learned under swap operations ################  

print("Start training")

epoch = 0

while epoch<1000:

    current_unitary = calc_unitary(mode_num, clements_mode, 
                                phi_list=params[:len(clements_mode)], theta_list=params[len(clements_mode):])
    
    temp_unitary = current_unitary @ torch.as_tensor(target_unitary)

    train_loss = 0
    for k in range(n_training):
        real_probs = torch.abs(calc_boson_sampling(temp_unitary, training_state_list[k], output_photon_basis))**2
        train_loss += torch.sum(hamming_dist_train[k]*real_probs)
    
    if epoch%100==0:
        print(train_loss.item(), optimizer.param_groups[0]["lr"])

    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if train_loss<1e-15:
        break

    epoch += 1
    scheduler.step(train_loss)


first_round_unitary = current_unitary.detach().numpy() @ target_unitary

for k in range(n_training):
    out_probs = calc_boson_sampling(first_round_unitary, training_state_list[k], 
                                    get_out_configs(mode_num, photon_num, if_bunching=True))
    # print(torch.abs(out_probs)**2)

################  Second round, find SWAP and train local phases seperately with a single photon ##################

# The second round is only needed for photon number >= 2
# First figuring out the correct swap by sending in one photon and checking from outcome probabilities 
# which swap gates need to be applied, then optimize final phases via 1-photon state

training_phase_prep_unitary = random_unitary(mode_num)

# prepared state before applying operations
probs_before = np.abs(training_phase_prep_unitary[:, 0])**2

# the output state after applying target unitary and trained unitary in the first round, which may have swapped modes
probs_after = np.abs(first_round_unitary @ training_phase_prep_unitary)[:, 0]**2

print(probs_before, probs_after)

if not np.allclose(np.sort(probs_before), np.sort(probs_after), atol=1e-4):
    print("Cannot find permutation, not enough training states for learning unitary in the first step")
    exit()


##swap different modes such that probabilties matches
swap_graph = np.zeros(mode_num,dtype=int)

for k in range(len(probs_before)):
    swap_graph[k] = np.argmin(np.abs(probs_before-probs_after[k]))

assert np.all(np.sort(swap_graph)==np.arange(mode_num))

print("Found permutation", swap_graph)

swap_permutations = np.zeros((mode_num, mode_num), dtype=np.complex128)
for i in range(mode_num):
    swap_permutations[swap_graph[i], i] = 1
        
first_round_unitary_swapped = np.dot(swap_permutations, first_round_unitary)


print("\nStart training phase by using single photon using trained circuit+SWAP")

epoch = 0

optimizer = optim.Adam([output_phase], lr=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30)

while epoch < 500:

    output_phase_unitary = torch.diag(torch.exp(1j*output_phase))
    temp_unitary = torch.as_tensor(training_phase_prep_unitary.T.conj()) @ output_phase_unitary @ torch.as_tensor(first_round_unitary_swapped @ training_phase_prep_unitary)

    real_probs = torch.abs(temp_unitary[:, 0])**2
    train_loss = torch.sum(real_probs[1:])

    if epoch%50==0:
        print(train_loss.item())

    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if train_loss<1e-15:
        break

    epoch += 1


final_unitary_phase = torch.diag(torch.exp(1j*output_phase))
final_unitary_full = final_unitary_phase.detach().numpy() @ first_round_unitary_swapped

print("\nNumber of total training data used", n_training+1)
print("i.e.",n_training,"Fock states","and 1 single-photon state")


######################## final testing after training #####################
# 1-photon states used for testing

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
print("Finished training, test error", error_test)
