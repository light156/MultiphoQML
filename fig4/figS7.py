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


x_unit=0.12
offset_list = [-x_unit*2,-x_unit,0,x_unit,x_unit*2]
color_list = ['#E64B35', '#F39B7F', '#91D1C2', '#4DBBD5', '#3C5488']
label_list = ['n=1','n=2','n=3','n=4','n=5']


for i, photon_num in enumerate(photon_num_list):
    
    for j, threshold in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
        save_key = f'{threshold:.2f}_photon_{photon_num}'
        temp = log_file[save_key]
        
        y_mean, error = tolerant_mean(temp)
        x, y, error = offset_list[i]+j, y_mean[-1], error[-1]
        if j<4:
            plt.errorbar(x,y,
                        color=color_list[i],
                        yerr=error,
                        capsize=3,
                        capthick=1.5,
                        elinewidth=1.5)
            plt.scatter(x,y, s=20, marker = "o", color=color_list[i])
        else:
            plt.errorbar(x,y, 
                        color=color_list[i],
                        yerr=error,
                        capsize=3,
                        capthick=1.5,
                        elinewidth=1.5,)
            plt.scatter(x,y, s=20, marker = "o", color=color_list[i],
                        label=label_list[i])

plt.xlabel('Angular distance margin')
plt.ylabel('Simulated test loss')

plt.xlim([-0.5, 4.5])
plt.xticks(range(5), ['0.5', '0.6', '0.7', '0.8', '0.9'])
plt.yscale('log')

plt.legend()
plt.show()