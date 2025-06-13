import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import json

from function import *
from process_dataset import vowel_dataset
from model import BQNN_simulation
from loss import PairwiseLoss, calc_pair_acc


def train(args, loss_fn, lr=0.1, epoch_num=300, save_dir=None, if_silent=True):
    
    train_data, test_data = vowel_dataset(train_ratio=0.7)

    x_train, y_train = torch.tensor(train_data[0]), torch.tensor(train_data[1])
    x_test, y_test = torch.tensor(test_data[0]), torch.tensor(test_data[1])

    model = BQNN_simulation(args)
    model.double()
    model_params = list(model.parameters())

    optimizer = optim.Adam(model_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20)

    train_index_loader = DataLoader(list(range(26)), batch_size=5, shuffle=True)

    ######### Save config and log file ###############
    if save_dir is None:
        save_dir = get_folder('vowel', prefix=args['mode'])
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/weights')

    with open(f'{save_dir}/args.json', 'w') as json_file: 
        json.dump(args, json_file)
    
    log = open(f'{save_dir}/log.csv', 'w')
    log.write(f'train_loss,test_loss,test_acc,learning_rate\n')

    model.eval()
    
    with torch.no_grad():
        out_train = model(x_train)
        train_loss = loss_fn(out_train, y_train).item()

        out_test = model(x_test)
        test_loss = loss_fn(out_test, y_test).item()
        test_acc = calc_pair_acc(out_test, y_test, threshold=loss_fn.threshold)
        
        log.write(f'{train_loss:.6f},{test_loss:.6f},{test_acc*100:.2f},{lr}\n')
        log.flush()

        if not if_silent:
            print(f'Before training')
            print(f'Train loss: {train_loss:>.6f}, Test loss: {test_loss:>.6f}, Test Acc: {(test_acc*100):.2f}%')
            print("-------------------------------")

    ############# Start training ################
    for epoch in range(1, epoch_num+1):
        
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < 5e-4:
            break

        # Training
        model.train()
        
        selected_index = next(iter(train_index_loader)).detach().numpy()
        slice_train_select = []
        for c in range(7):
            for num in selected_index:        
                slice_train_select.append(c*26+num)
        
        x, y = x_train[slice_train_select], y_train[slice_train_select]
        out_train = model(x)
        loss = loss_fn(out_train, y)

        train_loss = loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Evaluation
        model.eval()

        with torch.no_grad():
            out_test = model(x_test)
            test_loss = loss_fn(out_test, y_test).item()
            test_acc = calc_pair_acc(out_test, y_test, threshold=loss_fn.threshold)
            
            log.write(f'{train_loss:.6f},{test_loss:.6f},{test_acc*100:.2f},{current_lr}\n')
            log.flush()

            if not if_silent:
                print(f'Epoch {epoch}, Learning rate: {current_lr}')
                print(f'Train loss: {train_loss:>.6f}, Test loss: {test_loss:>.6f}, Test Acc: {(test_acc*100):.2f}%')
                print("-------------------------------")
                
        torch.save(model.state_dict(), f'{save_dir}/weights/model_{epoch:03d}.pth')
        scheduler.step(train_loss)

    log.close()


if __name__ == '__main__':

    for threshold in [0.90]:

        loss_fn = PairwiseLoss(threshold, margin=0, mode='all', if_mean=True)

        in_config = [1, 1, 0, 0, 0, 0]

        seed = 0
        args = {'threshold': threshold,
                'in_config': in_config,
                'out_config_list': get_out_configs(photon_num=sum(in_config), mode_num=6)}

        save_dir = f'metric_learning/results/{threshold:.2f}_photon_{sum(in_config)}/seed_{seed}'
        
        torch.manual_seed(seed)
        train(args, loss_fn, lr=0.1, epoch_num=100, save_dir=save_dir, if_silent=False)

