import numpy as np
import torch


class SPSA:

    def __init__(self, c, a, A=0, alpha=0.602, gamma=0.101, start_epoch=0):
        self.c = c
        self.a = a
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

        self.k = start_epoch
        self.c_k = self.c / ((self.k+1)**self.gamma)
        self.a_k = self.a / ((self.A+self.k+1)**self.alpha)
        

    def calibrate(self, model, params_to_update, loss_fn, x, y): 

        target_magnitude = 2 * np.pi / 10

        avg_magnitudes = 0.0
        steps = 10

        for _ in range(steps):

            self.random_vec_list = []
            for p in params_to_update:
                assert p.requires_grad is True
                # Rademacher distribution, all -1 or 1
                random_binary = torch.randint(0, 2, p.shape)*2-1
                self.random_vec_list.append(random_binary)

            out_pos, out_neg = self.forward(model, params_to_update, x)
            loss_pos, loss_neg = loss_fn(out_pos, y), loss_fn(out_neg, y)
            
            delta = loss_pos - loss_neg
            avg_magnitudes += torch.abs(delta / (2 * self.c_k))

        avg_magnitudes /= steps
        
        a = target_magnitude / avg_magnitudes

        # compute the rescaling factor for correct first learning rate
        if a < 1e-10:
            print(f"Calibration failed, using {target_magnitude} for `a`")
            a = target_magnitude

        print("Finished calibration: Learning rate a = %s" % a)
        return a


    def set_random_perturbation(self, params_to_update):

        self.random_vec_list = []

        for p in params_to_update:
            assert p.requires_grad is True
            # Rademacher distribution, all -1 or 1
            random_binary = torch.randint(0, 2, p.shape)*2-1
            self.random_vec_list.append(random_binary)
        
        
    def forward(self, model, params_to_update, x):

        for p, random_binary in zip(params_to_update, self.random_vec_list):
            p.data += self.c_k*random_binary
            # print(p)

        out_pos = model(x)

        for p, random_binary in zip(params_to_update, self.random_vec_list):
            p.data -= self.c_k*random_binary*2
            # print(p)

        out_neg = model(x)

        for p, random_binary in zip(params_to_update, self.random_vec_list):
            p.data += self.c_k*random_binary
            # print(p)

        return out_pos, out_neg


    def step(self, params_to_update, loss_pos, loss_neg):

        const = (loss_pos-loss_neg)/2/self.c_k

        for p, random_binary in zip(params_to_update, self.random_vec_list):
            p.grad = const/random_binary
            p.data -= self.a_k * p.grad

        self.k += 1
        self.c_k = self.c / ((self.k+1)**self.gamma)
        self.a_k = self.a / ((self.A+self.k+1)**self.alpha)


    def zero_grad(self, params_to_update):
        
        for p in params_to_update:
            p.grad = torch.zeros_like(p.data)