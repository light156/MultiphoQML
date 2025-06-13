import numpy as np
import matplotlib.pyplot as plt
    

photon_num_list = [1, 2, 3, 4, 5]

log_file = np.load('fig4/sim_all_logs.npy', allow_pickle=True).item()


def tolerant_mean(arrs, dim=1, epoch_min=300):
    lens = [len(i) for i in arrs]
    arr = np.zeros((epoch_min,len(arrs)))
    for idx, l in enumerate(arrs):
        if len(l) < epoch_min:
            arr[:len(l),idx] = l[:, dim]
            arr[len(l):,idx] = arr[len(l)-1,idx]
        else:
            arr[:,idx] = l[:epoch_min, dim]
            
    return arr.mean(axis = -1), arr.std(axis=-1)


threshold = 0.9

color_list = ['#E64B35', '#F39B7F', '#91D1C2', '#4DBBD5', '#3C5488']
label_list = ['n=1','n=2','n=3','n=4','n=5']


for photon_num in photon_num_list:
    
    save_key = f'{threshold:.2f}_photon_{photon_num}'
    temp = log_file[save_key]

    y, error = tolerant_mean(temp)
    plt.plot(y, color=color_list[photon_num-1], label=label_list[photon_num-1])
    plt.fill_between(np.arange(len(y)), y-error, y+error, color=color_list[photon_num-1], alpha=0.15, linewidth=0)
    #print(photon_num, y[0])


plt.ylabel('Sim test loss')
plt.xlabel('Epoch')
plt.xlim(0,300)
plt.yscale('log')

plt.legend()
plt.show()