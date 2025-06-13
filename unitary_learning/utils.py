import numpy as np
from sympy.utilities.iterables import multiset_permutations


##had to reimpliment the beam splitter because the original function uses np.complex64
def beam_splitter_clements(
    theta: float, phi: float, dim: int, mode_1: int, mode_2: int
):
    """Create the beam splitter matrix with reflectivity cos(θ) and phase shift φ
    acting on mode_1 and mode_2. Follows Clemens et al. [1].

    References:
        [1] Clements et al., "An Optimal Design for Universal Multiport
            Interferometers" arXiv, 2007. https://arxiv.org/pdf/1603.08788.pdf
            
            
    """
    ##had to reimpliment the beam splitter because the original function uses np.complex64
    T = np.eye(dim, dtype=np.complex128)
    T[mode_1, mode_1] = 1j * np.exp(1j*theta/2) * (np.exp(1j*phi) * np.sin(theta/2))
    T[mode_1, mode_2] = 1j * np.exp(1j*theta/2) * (np.cos(theta/2))
    T[mode_2, mode_1] = 1j * np.exp(1j*theta/2) * (np.exp(1j*phi) * np.cos(theta/2))
    T[mode_2, mode_2] = 1j * np.exp(1j*theta/2) * (-1 * np.sin(theta/2))

    return T


def get_gate_positions(n_gates,modes):
    gate_positions=[[2*j,(2*j+1)%modes] for j in range(modes//2)]+[[2*j+1,(2*j+2)%modes] for j in range((modes-1)//2)]
    gate_positions=gate_positions*((2*n_gates)//modes)
    gate_positions=gate_positions[:n_gates]

    return gate_positions


def total_swap_gate(dim,mode_ini,mode_out):
    T = np.zeros([dim,dim], dtype=np.complex128)
    for i in range(len(mode_ini)):
        T[mode_out[i], mode_ini[i]] = 1
        
    return T


def get_unitary(parameter_list,modes,gate_positions):
    lin_unitary=np.eye(modes,dtype=np.complex128)

    for q in range(len(gate_positions)):
        theta=parameter_list[2*q]
        phi=parameter_list[2*q+1]
        pos1,pos2=gate_positions[q]
        
        bs_matrix=beam_splitter_clements(theta=theta, phi=phi, dim=modes, mode_1=pos1, mode_2=pos2)
        lin_unitary=np.dot(bs_matrix,lin_unitary)

    return lin_unitary


def get_unitary_clements(parameter_list,modes):

    gate_positions=get_gate_positions(modes*(modes-1)//2,modes)
    lin_unitary=np.eye(modes,dtype=np.complex128)
    
    for q in range(len(gate_positions)):
        theta=parameter_list[2*q]
        phi=parameter_list[2*q+1]
        pos1,pos2=gate_positions[q]
        
        bs_matrix=beam_splitter_clements(theta=theta, phi=phi, dim=modes, mode_1=pos1, mode_2=pos2)
        lin_unitary=np.dot(bs_matrix,lin_unitary)
    
    ext_phi = parameter_list[2*len(gate_positions):]
    ext_phi_matrix=np.diag(np.exp(-1j*np.array(ext_phi,dtype=np.complex128)))
    lin_unitary=np.dot(ext_phi_matrix,lin_unitary)

    return lin_unitary


def accel_asc(n):
    """
    Generates Integer partitions of any positive integer, i.e.
    all positive integer sequences that sum up to the integer n.
    obtained from: (https://jeromekelleher.net/generating-integer-
    partitions.html).

    Yields:
        [int]:
            A list of integers that have a sum equal to n
    """
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]


def get_out_configs(mode_num, photon_num, if_bunching: bool = False):
    """
    Args:
        mode_num:
        photon_num:
        if_bunching: whether to keep bunching configurations in outputs

    Returns:
        A list with all possible configurations given the photon number and port number
    """
    # generate all possible photon outputs that sum up to the number of photons inputted.
    # Ex: 5 photon input can produce an output of [5], [4,1], [3,2],...[1,1,1,1,1]
    # Note that the notation above does NOT account for mode limitations
    # so Kelleher's function (see `accel_asc()``) may generate a
    # configuration that is not physically feasible (exceeds number of available modes).
    # The `len(num_photons_partition <= num_modes)` ensures this does not happen.
    config_list = []

    for num_photons_partition in accel_asc(photon_num):
        if len(num_photons_partition) <= mode_num:
            # certain outputs may fall shy of the number of modes
            # and need to be zero-padded
            zero_arr = [0] * mode_num
            for idx, val in enumerate(num_photons_partition):
                zero_arr[idx] = val
            # multiset_permutation usage to generate unique permutations of a list:
            # (https://stackoverflow.com/a/41210450)
            # for each properly padded output, generate all unique permutations of that output.
            # Ex: [1,0,0,0] can produced [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1].
            for p in list(multiset_permutations(zero_arr))[::-1]:
                config_list.append(p)

    if not if_bunching:
        config_list = [c for c in config_list if all(i <= 1 for i in c)]

    return config_list


def calc_bs_two_photons(U, in_ports, output_photon_basis, identical_photon=True):
    
    # in ports index start from 1
    assert in_ports[0] != in_ports[1]

    out_probs = []

    for out_ports in output_photon_basis:
        if identical_photon == True:
            single_prob = np.abs(U[out_ports[0]-1, in_ports[0]-1]*U[out_ports[1]-1, in_ports[1]-1]+\
                                 U[out_ports[1]-1, in_ports[0]-1]*U[out_ports[0]-1, in_ports[1]-1])**2
        else:
            single_prob = np.abs(U[out_ports[0]-1, in_ports[0]-1])**2 * np.abs(U[out_ports[1]-1, in_ports[1]-1])**2+\
                          np.abs(U[out_ports[1]-1, in_ports[0]-1])**2 * np.abs(U[out_ports[0]-1, in_ports[1]-1])**2
        
        if out_ports[0] == out_ports[1]:
            single_prob /= 2

        out_probs.append(single_prob)

    return np.array(out_probs)


if __name__ == '__main__':
    print(get_gate_positions(5, 2))
    print(get_gate_positions(6, 4))
    print(get_gate_positions(10, 5))
    print(get_gate_positions(15, 6))
