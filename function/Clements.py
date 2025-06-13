import numpy as np
import torch
import scipy as sp
import random


def calc_unitary(mode_num, modes_list, theta_list, phi_list):
    assert len(modes_list) == len(phi_list) and len(modes_list) == len(theta_list), \
        "Mode pair number should be equal with the phase number"

    U = torch.eye(mode_num, dtype=torch.complex128)

    for modes, theta, phi in zip(modes_list, theta_list, phi_list):
        T = torch.eye(mode_num, dtype=torch.complex128)
        
        factor = 1j * torch.exp(1j * theta/2)
        T[modes[0] - 1, modes[0] - 1] = torch.exp(1j * phi) * torch.sin(theta/2) * factor
        T[modes[0] - 1, modes[1] - 1] = torch.cos(theta/2) * factor
        T[modes[1] - 1, modes[0] - 1] = torch.exp(1j * phi) * torch.cos(theta/2) * factor
        T[modes[1] - 1, modes[1] - 1] = -torch.sin(theta/2) * factor

        U = T @ U

    return U


def calc_unitary_numpy(mode_num, modes_list, theta_list, phi_list):
    assert len(modes_list) == len(phi_list) and len(modes_list) == len(theta_list), \
        "Mode pair number should be equal with the phase number"

    U = np.eye(mode_num, dtype=np.complex128)

    for modes, theta, phi in zip(modes_list, theta_list, phi_list):
        T = np.eye(mode_num, dtype=np.complex128)
        
        factor = 1j * np.exp(1j * theta/2)
        T[modes[0] - 1, modes[0] - 1] = np.exp(1j * phi) * np.sin(theta/2) * factor
        T[modes[0] - 1, modes[1] - 1] = np.cos(theta/2) * factor
        T[modes[1] - 1, modes[0] - 1] = np.exp(1j * phi) * np.cos(theta/2) * factor
        T[modes[1] - 1, modes[1] - 1] = -np.sin(theta/2) * factor

        U = T @ U

    return U


def make_T(theta, phi):
    T = np.eye(2, dtype=np.complex_)
    T[0,0] = 1j * np.exp(1j*theta/2) * (np.exp(1j*phi) * np.sin(theta/2))
    T[0,1] = 1j * np.exp(1j*theta/2) * (np.cos(theta/2))
    T[1,0] = 1j * np.exp(1j*theta/2) * (np.exp(1j*phi) * np.cos(theta/2))
    T[1,1] = 1j * np.exp(1j*theta/2) * (-1 * np.sin(theta/2))
    return T


def rearrange_clements(output):
    N = len(output) 
    output = output.copy() 
    for i in range(1,N-1):
        current_input = output[i][0]
        next_input = output[i+1][0]
        if current_input % 2 == 1:
            if current_input < N-1:
                if next_input == current_input +2:
                    continue 
                else:
                    for j in range(i+1, len(output)-1): 
                        if output[j][0] == current_input +2:
                            output.insert(i+1, output.pop(j))
                            break 
        else:
            if current_input < N-2:
                if next_input == current_input +2:
                    continue
                else:
                    for j in range(i+1, len(output)-1):
                        if output[j][0] == current_input +2:
                            output.insert(i+1, output.pop(j))
                            break
    return output


def square_decomposition(U, rearrange = True):
    N = int(np.sqrt(U.size))
    left_T = []
    BS_list = []
    
    for ii in range(N-1):
        if np.mod(ii,2) == 0:
            for jj in range(ii+1):
                modes = [ii-jj+1,ii+2-jj]
                a = U[N-1-jj,ii-jj]
                b = U[N-1-jj,ii-jj+1]
                if np.around(abs(a),4) != 0:
                    theta = 2*(np.angle(abs(a) + 1j*abs(b)))
                else: theta = np.pi
                if abs(np.around(a,4)) != 0 and abs(np.around(b,4)) != 0:
                    a_ang = np.angle(np.around(U,5)[N-1-jj,ii-jj])
                    b_ang = np.angle(np.around(U,5)[N-1-jj,ii-jj+1])
                    phi = a_ang - b_ang + np.pi
                    if phi < 0: phi = phi + 2*np.pi
                else: phi = 0
                invT = np.eye(N,dtype=np.complex_)
                invT[modes[0]-1:modes[1],modes[0]-1:modes[1]] = make_T(theta,phi).conj().T
                U = np.matmul(U,invT)
                BS_list.append([modes[0],modes[1],theta,phi])
        else:
            for jj in range(ii+1):
                modes = [N+jj-ii-1,N+jj-ii]
                a = U[N+jj-ii-2,jj]
                b = U[N+jj-ii-1,jj]
                if np.around(abs(b),4) != 0:
                    theta = 2*(np.angle(1j*abs(a) + abs(b)))
                else: theta = np.pi
                if abs(np.around(a,4)) != 0 and abs(np.around(b,4)) != 0:
                    a_ang = np.angle(np.around(U,5)[N+jj-ii-2,jj])
                    b_ang = np.angle(np.around(U,5)[N+jj-ii-1,jj])
                    phi = b_ang - a_ang
                    if phi < 0: phi = phi + 2*np.pi
                else: phi = 0
                T = np.eye(N,dtype=np.complex_)
                T[modes[0]-1:modes[1], modes[0]-1:modes[1]] = make_T(theta,phi)
                U = np.matmul(T,U)
                left_T.append([modes[0],modes[1],theta,phi])
    for BS in np.flip(left_T,0):
        modes = [int(BS[0]),int(BS[1])]
        invT = np.eye(N,dtype=np.complex_)
        invT[modes[0]-1:modes[1],modes[0]-1:modes[1]] = make_T(BS[2],BS[3]).conj().T
        U = np.matmul(invT,U)
        a = U[modes[1]-1,modes[0]-1]
        b = U[modes[1]-1,modes[1]-1]
        if np.around(abs(a),4) != 0:
            theta = 2*(np.angle(abs(a) + 1j*abs(b)))
        else: theta = np.pi
        if abs(np.around(a,4)) != 0 and abs(np.around(b,4)) != 0:
            a_ang = np.angle(np.around(U,5)[modes[1]-1,modes[0]-1])
            b_ang = np.angle(np.around(U,5)[modes[1]-1,modes[1]-1])
            phi = a_ang - b_ang + np.pi
            if phi < 0: phi = phi + 2*np.pi
        else: phi = 0
        invT[modes[0]-1:modes[1],modes[0]-1:modes[1]] = make_T(theta,phi).conj().T
        U = np.matmul(U,invT)
        BS_list.append([modes[0],modes[1],theta,phi])

    phases = np.diag(U)
    output_phases = [np.angle(i) for i in phases]
    
    if rearrange:
        BS_list = rearrange_clements(BS_list)
    
    modes_list = [(BS[0], BS[1]) for BS in BS_list]
    theta_list = [BS[2] for BS in BS_list]
    phi_list = [BS[3] for BS in BS_list]
    
    return modes_list, theta_list, phi_list, output_phases


def clements_mode_list(mode_num):

    mode_list = []
    
    for k in range(mode_num):
        if k%2==0:
            for i in range(1, mode_num, 2):
                mode_list.append((i, i+1))
        else:
            for i in range(2, mode_num, 2):
                mode_list.append((i, i+1))

    return mode_list


def random_modes_list(mode_num, mzi_num):
    modes_list = [[i, j] for i in range(1, mode_num) for j in range(i+1, mode_num+1)]
    last_choice = None
    n = 0

    temp_modes = []
    while n < mzi_num:
        new_choice = random.choice(modes_list)
        if new_choice != last_choice:
            last_choice = new_choice
            n += 1
            temp_modes.append(new_choice)

    return temp_modes


def random_unitary(N, real=False):
    r"""Random unitary matrix representing an interferometer.
    Copy from strawberryfields, For more details, see :cite:`mezzadri2006`.

    Args:
        N (int): number of modes
        real (bool): return a random real orthogonal matrix

    Returns:
        array: random :math:`N\times N` unitary distributed with the Haar measure
    """
    if real:
        z = np.random.randn(N, N)
    else:
        z = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2.0)

    q, r = sp.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    U = np.multiply(q, ph, q)
    return U



if __name__ == '__main__':

    # for i in [4,5,6,7,8]:
    #     print(clements_mode_list(i))

    for mode_num in [4,5,6,7,8]:
        print(mode_num)

        for _ in range(10):
            U = random_unitary(mode_num)
            modes_list, theta_list, phi_list, output_phases = square_decomposition(U)
            U1 = calc_unitary(mode_num, modes_list, torch.tensor(theta_list), torch.tensor(phi_list)).numpy()
            U2 = np.diag(np.exp(1j*np.array(output_phases)))

            assert np.allclose(U, U2@U1, atol=1e-5)
