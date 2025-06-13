import numpy as np
import torch

chs = [1,2,3,4,5,6,7,8]

chs_patterns = [(chs[i],chs[j]) for i in range(8) for j in range(8) if i < j]
fiber2det = {2: [1, 2], 3: [3, 6], 4: [5], 5: [4, 8], 6: [7]}
# bunching and not bunching patterns
pat_bunch = {(f,f):tuple(d) for f,d in fiber2det.items() if len(d) == 2}
pat_notbunch = { (f1,f2): [ (c1, c2) if c1 < c2 else (c2, c1)
                            for c1 in d1 for c2 in d2 if c1!=c2 ] 
                for f1,d1 in fiber2det.items() for f2,d2 in fiber2det.items() if f1 < f2}

a = np.load('unitary_hardware/cali_8_detectors.npy', allow_pickle=True)

fid_true_list, fid_false_list = [], []

weight = torch.ones(9, dtype=torch.float64, requires_grad=True)
# weight = torch.tensor([1,1,1,2,1,1,1,2], dtype=torch.float64, requires_grad=True)
# weight[3] = 1.8
# weight[7] = 1.8

optimizer = torch.optim.Adam([weight], lr=0.1)


def temp_func(cc):
    cc_dict = dict(zip(chs_patterns, cc))

    nobun = [torch.stack([cc_dict[pp]*(weight[pp[0]-1]**2)*(weight[pp[1]-1]**2) for pp in p]).sum() for f, p in pat_notbunch.items()]
    bun = [cc_dict[p]*weight[8]*(weight[p[0]-1]**2)*(weight[p[1]-1]**2) for f, p in pat_bunch.items()]
    
    data = torch.concat([torch.stack(nobun), torch.stack(bun)])
    # print(data)
    bs_exp = data/torch.sum(data)
    return bs_exp


for epoch in range(100):

    if epoch%10==0:

        fid_true_list = []
        for i in range(100):
            U_theo, res34, res56 = a[i]

            cc, bs_theo_true, bs_theo_false = res34
            bs_exp = temp_func(cc)
            fid_true_list.append(torch.sum(torch.sqrt(bs_exp*torch.as_tensor(bs_theo_true))))
        
        fid_true_avg = torch.sum(torch.stack(fid_true_list))/100
        print(epoch, fid_true_avg)


    fid_true_sum = 0

    for i in range(20):
        U_theo, res34, res56 = a[i]

        cc, bs_theo_true, _ = res34
        # print(bs_theo_true)
        bs_exp = temp_func(cc)
        fid_true_sum += torch.sum(torch.sqrt(bs_exp*torch.as_tensor(bs_theo_true)))

    
    for i in range(20):
        U_theo, res34, res56 = a[i]

        cc, bs_theo_true, _ = res56
        # print(bs_theo_true)
        bs_exp = temp_func(cc)
        fid_true_sum += torch.sum(torch.sqrt(bs_exp*torch.as_tensor(bs_theo_true)))

    loss = 1-fid_true_sum/40
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()



fid_true_list = []
for i in range(100):
    U_theo, res34, res56 = a[i]

    cc, bs_theo_true, bs_theo_false = res34
    bs_exp = temp_func(cc)
    fid_true_list.append(torch.sum(torch.sqrt(bs_exp*torch.as_tensor(bs_theo_true))))

fid_true_list = torch.stack(fid_true_list)
print(torch.min(fid_true_list), torch.mean(fid_true_list), torch.max(fid_true_list))


fid_true_list = []
for i in range(100):
    U_theo, res34, res56 = a[i]

    cc, bs_theo_true, bs_theo_false = res56
    bs_exp = temp_func(cc)
    fid_true_list.append(torch.sum(torch.sqrt(bs_exp*torch.as_tensor(bs_theo_true))))

fid_true_list = torch.stack(fid_true_list)
print(torch.min(fid_true_list), torch.mean(fid_true_list), torch.max(fid_true_list))



print(weight)
