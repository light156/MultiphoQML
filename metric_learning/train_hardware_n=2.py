import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

from function import *
from process_dataset import vowel_dataset
from loss import calc_pair_acc, PairwiseLoss
from model import BQNN_hardware


threshold = 0.90

train_data, test_data = vowel_dataset(train_ratio=0.7)

x_train, y_train = torch.tensor(train_data[0]), torch.tensor(train_data[1])
x_test, y_test = torch.tensor(test_data[0]), torch.tensor(test_data[1])

############################################
seed = 2
torch.manual_seed(seed)

args = {'in_config': [1, 0, 0, 0, 0, 1],
        'out_config_list': get_out_configs(photon_num=2, mode_num=6, if_bunching=False),
        'photon_num': 2}

model = BQNN_hardware(args)
model.double()
model_params = list(model.parameters())

loss_fn = PairwiseLoss(threshold, margin=0, mode='all')

######### Save config and log file ###############

save_dir = get_folder('metric_learning', prefix=f'hardware_n=2_{threshold:.2f}_seed_{seed}')
# os.makedirs(save_dir+'/exp_data')

log = open(f'{save_dir}/log.csv', 'w')
log.write(f'train_loss,test_loss,train_acc,test_acc\n')

################# Experiment Start ######################

chip_phase_array = np.zeros((15, 3))
chip_phase_array[:, 0] = np.arange(15)
mzi_idx = [0] + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

chip_mzi_list = clements_mode_list(mode_num=6)

def fake_real_experiment(batch_theta_phi_list):
    exp_bs_probs_list = []

    for theta_list, phi_list in batch_theta_phi_list:
        
        chip_phase_array[mzi_idx, 1] = np.array(torch.tensor(theta_list))
        chip_phase_array[mzi_idx, 2] = np.array(torch.tensor(phi_list))
        
        ########## replace experiment code ################
        # set_all_phase(chip_phase_array)
        # time.sleep(2) # chip setting time
        # data = measure(server, step=1)['cc'][0]
        # out_probs = data/np.sum(data)
        
        unitary = calc_unitary(6, chip_mzi_list, 
                               theta_list=torch.as_tensor(chip_phase_array[:, 1]), 
                               phi_list=torch.as_tensor(chip_phase_array[:, 2]))
        
        complex_amplitude = calc_boson_sampling(unitary, args['in_config'], args['out_config_list'])
        out_probs = torch.abs(complex_amplitude) ** 2
        out_probs = F.normalize(out_probs, p=1, dim=0).detach().numpy()

        exp_bs_probs_list.append(torch.sqrt(torch.tensor(out_probs)))
        ###################################################

    return torch.stack(exp_bs_probs_list)


train_index_loader = DataLoader(list(range(26)), batch_size=5, shuffle=True)

# code for continuing the hardware experiment, in case that the experiment may be terminated by unforced issues
start_step = 0
# start_folder = 'vowel_hardware/hardware_0730_115627'
# model.load_state_dict(torch.load(f'{start_folder}/weights/model_{start_step:03d}.pth'))

spsa = SPSA(a=150, c=0.4, start_epoch=start_step)

# fix the behaviour of random seeds by resampling the previous training process
for epoch in range(1, start_step+1):
    spsa.set_random_perturbation(model_params)
    selected_index = next(iter(train_index_loader)).detach().numpy()


def train(step, slice_train_select):
    
    x, y = x_train[slice_train_select], y_train[slice_train_select]

    model.train()

    spsa.set_random_perturbation(model_params)
    random_vec_list = spsa.random_vec_list
    # print(spsa.random_vec_list)

    ################# simulation #####################
    sim_complex_probs_pos, sim_complex_probs_neg = spsa.forward(model.calc_state, model_params, x)
    sim_bs_probs_pos, sim_bs_probs_neg = torch.abs(sim_complex_probs_pos), torch.abs(sim_complex_probs_neg)
    # print(nn.CosineSimilarity()(torch.abs(sim_complex_probs_pos), torch.abs(sim_complex_probs_neg)))

    ################# experiment ################
    for p, random_binary in zip(model_params, random_vec_list):
        p.data += spsa.c_k*random_binary

    batch_theta_phi_list_pos = model.get_batch_ansatz(x)

    exp_bs_probs_pos = fake_real_experiment(batch_theta_phi_list_pos)

    for p, random_binary in zip(model_params, random_vec_list):
        p.data -= spsa.c_k*random_binary*2

    batch_theta_phi_list_neg = model.get_batch_ansatz(x)

    exp_bs_probs_neg = fake_real_experiment(batch_theta_phi_list_neg)

    for p, random_binary in zip(model_params, random_vec_list):
        p.data += spsa.c_k*random_binary

    sim_exp_prob_dict = {'sim_pos': sim_bs_probs_pos, 'sim_neg': sim_bs_probs_neg,
                         'exp_pos': exp_bs_probs_pos, 'exp_neg': exp_bs_probs_neg,
                         'slice_train_select': slice_train_select}

    # np.save(f'{save_dir}/exp_data/step_{step:03d}_train.npy', sim_exp_prob_dict)

    exp_loss_pos, exp_loss_neg = loss_fn(exp_bs_probs_pos, y), loss_fn(exp_bs_probs_neg, y)
    spsa.step(model_params, exp_loss_pos, exp_loss_neg)


def eval(step):
    model.eval()

    with torch.no_grad():

        ############### train dataset ##################### 
        out_train = model(x_train)
        train_loss = loss_fn(out_train, y_train).item()
        train_acc = calc_pair_acc(out_train, y_train, threshold)

        out_test = model(x_test)
        test_loss = loss_fn(out_test, y_test).item()
        test_acc = calc_pair_acc(out_test, y_test, threshold)
        
        torch.save(model.state_dict(), f'{save_dir}/weights/model_{step:03d}.pth')
        
        print(f'Train loss: {train_loss:>.6f}, Train Acc: {(train_acc*100):.2f}%')
        print(f'Test loss: {test_loss:>.6f}, Test Acc: {(test_acc*100):.2f}%')
        print("-------------------------------")        
        
        """
        matrix = torch.matmul(out_test, out_test.T).detach().numpy()
        norm = TwoSlopeNorm(vcenter=threshold, vmin=min(np.min(matrix), 2*threshold-1), vmax=1.0)

        plt.imshow(matrix, cmap='RdBu_r', norm=norm)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Step {step:03d}, Test acc: {test_acc*100:.2f}%')
        plt.colorbar()
        plt.savefig(f'{save_dir}/step_{step:03d}.png')
        plt.clf()

        log.write(f'{train_loss:.6f},{test_loss:.6f},{train_acc*100:.2f},{test_acc*100:.2f}\n')
        log.flush()
        """


print('Before training')
eval(step=start_step)

for step in range(start_step+1, 201):
    print(f"Step {step}: a_k = {spsa.a_k}, c_k = {spsa.c_k}")

    selected_index = next(iter(train_index_loader)).detach().numpy()

    slice_train_select = []
    for c in range(7):
        for num in selected_index:        
            slice_train_select.append(c*26+num)
    
    # Training
    train(step, slice_train_select)

    # Evaluation
    eval(step)
    
