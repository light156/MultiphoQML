import scipy as sp
from math import sqrt, pi
from sympy.utilities.iterables import multiset_permutations
from function.BS_utils import *


def calc_boson_sampling(unitary, in_config, out_config_list):
    """
    Args:
        unitary: the matrix
        in_config_list: list like [2, 0, 1, 0], indicating the photon number for each port
        out_config_list: list of different possible configs, like [[0, 1, 1, 1], [0, 1, 2, 0], [1, 1, 0, 1]]

    Returns:
        Complex probability vector for all the output configurations
    """
    unitary = torch.as_tensor(unitary)
    output_vec = torch.zeros(len(out_config_list), dtype=unitary.dtype)

    for i, out_config in enumerate(out_config_list):
        sub_mat = get_submatrix(unitary, in_config, out_config)
        denom = (sp.special.factorial(in_config).prod() *
                 sp.special.factorial(out_config).prod())
        output_vec[i] = perm_recursive(sub_mat) / sqrt(denom)

    # print(torch.sum(torch.abs(output_vec)**2))
    return output_vec


def calc_boson_sampling_multi_input(unitary, in_config_list, out_config_list, weight_list=None):
    """
    Args:
        unitary: the matrix
        in_config_list: list like [[2, 0, 1, 0], [1, 1, 0, 1]], indicating the photon number for each port
        weight_list: list like [1/sqrt(2), 1j/sqrt(2)], indicating the weight for each input configuration
        out_config_list: list of different possible configs, like [[0, 1, 1, 1], [0, 1, 2, 0], [1, 1, 0, 1]]

    Returns:
        Complex probability vector for all the output configurations
    """

    if weight_list is None:
        weight_list = [1/sqrt(len(in_config_list))] * len(in_config_list)
    else:
        assert len(in_config_list) == len(weight_list), "The length of in_config_list and weight_list should be the same"
        assert np.isclose(np.sum(np.abs(weight_list)**2), 1), "The weight list should be normalized to 1"

    unitary = torch.as_tensor(unitary)
    output_vec = torch.zeros(len(out_config_list), dtype=unitary.dtype)

    for in_config, weight in zip(in_config_list, weight_list):
        for i, out_config in enumerate(out_config_list):
            sub_mat = get_submatrix(unitary, in_config, out_config)
            denom = (sp.special.factorial(in_config).prod() *
                     sp.special.factorial(out_config).prod())
            output_vec[i] += perm(sub_mat) * weight / sqrt(denom)

    # print(torch.sum(torch.abs(output_vec)**2))
    return output_vec


"""
def calc_distinguishable_photons(unitary, in_config: List[int]):
    col_list = []
    for col, val in enumerate(in_config):
        for _ in range(val):
            col_list.append(unitary[:, col])

    output_vec = col_list[0]

    for i in range(1, len(col_list)):
        temp = torch.outer(output_vec, col_list[i])
        output_vec = temp.flatten()

    return output_vec
"""


def get_submatrix(matrix, in_config, out_config):
    if sum(in_config) != sum(out_config):
        raise ValueError("Number of photons inputted is not equal"
                         "to number outputted!")

    photon_count = sum(in_config)

    # Row dimension is preserved but the number of columns is equal to the number of photons.
    col_mat = torch.zeros((matrix.shape[0], photon_count), dtype=matrix.dtype)

    # Given a photon input [1,2,0,0], the first column of the main unitary matrix is copied,
    # then the second column is copied TWICE with no other columns copied to create an (N x 3) submatrix
    col_idx = 0
    for col, val in enumerate(in_config):
        for i in range(val):
            col_mat[:, col_idx] = matrix[:, col]
            col_idx += 1

    # Create the final submatrix
    sub_mat = torch.zeros((photon_count, photon_count), dtype=matrix.dtype)
    # If the output is [1,2,0,0] then the first row of `col_mat` is
    # copied once, then the second row copied TWICE to create the final submatrix
    row_idx = 0
    for row, val in enumerate(out_config):
        for i in range(val):
            sub_mat[row_idx, :] = col_mat[row, :]
            row_idx += 1

    return sub_mat


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


if __name__ == '__main__':

    from function import *
    torch.manual_seed(0)

    mode_num = 6
    mzi_total_num = mode_num*(mode_num-1)//2
    modes_list = clements_mode_list(mode_num)

    prep_unitary = torch.as_tensor(random_unitary(mode_num))

    # Bell state
    in_config_list = [[1,0,1,0,0,0], [0,1,0,1,0,0]]
    output_config_list = get_out_configs(mode_num, photon_num=2, if_bunching=True)

    param_theta = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    param_phi = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    trainable_unitary = calc_unitary(mode_num, modes_list, param_theta, param_phi)

    state = calc_boson_sampling_multi_input(trainable_unitary@prep_unitary, in_config_list, output_config_list)
    qfi_matrix = calc_qfi_matrix(state, (param_theta, param_phi))
    eig, _ = np.linalg.eig(np.asarray(qfi_matrix))   
    eig_num = np.sum(eig>1e-6)
    print(eig_num)

    # "Bell state" with different weights
    in_config_list = [[1,0,1,0,0,0],[0,1,0,1,0,0]]
    output_config_list = get_out_configs(mode_num, photon_num=2, if_bunching=True)
    weight_list = [1/sqrt(5), 2/sqrt(5)]

    param_theta = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    param_phi = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    trainable_unitary = calc_unitary(mode_num, modes_list, param_theta, param_phi)

    state = calc_boson_sampling_multi_input(trainable_unitary@prep_unitary, in_config_list, output_config_list, weight_list)
    qfi_matrix = calc_qfi_matrix(state, (param_theta, param_phi))
    eig, _ = np.linalg.eig(np.asarray(qfi_matrix))   
    eig_num = np.sum(eig>1e-6)
    print(eig_num)

    # GHZ state
    in_config_list = [[1,1,1,0,0,0], [0,0,0,1,1,1]]
    output_config_list = get_out_configs(mode_num, photon_num=3, if_bunching=True)

    param_theta = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    param_phi = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    trainable_unitary = calc_unitary(mode_num, modes_list, param_theta, param_phi)

    state = calc_boson_sampling_multi_input(trainable_unitary@prep_unitary, in_config_list, output_config_list)
    qfi_matrix = calc_qfi_matrix(state, (param_theta, param_phi))
    eig, _ = np.linalg.eig(np.asarray(qfi_matrix))   
    eig_num = np.sum(eig>1e-6)
    print(eig_num)

    # W state
    in_config_list = [[0,1,1,0,1,0], [1,0,0,1,1,0], [1,0,1,0,0,1]]
    output_config_list = get_out_configs(mode_num, photon_num=3, if_bunching=True)

    param_theta = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    param_phi = torch.rand(mzi_total_num, dtype=torch.float64, requires_grad=True) * pi * 2
    trainable_unitary = calc_unitary(mode_num, modes_list, param_theta, param_phi)
    
    state = calc_boson_sampling_multi_input(trainable_unitary@prep_unitary, in_config_list, output_config_list)
    qfi_matrix = calc_qfi_matrix(state, (param_theta, param_phi))
    eig, _ = np.linalg.eig(np.asarray(qfi_matrix))   
    eig_num = np.sum(eig>1e-6)
    print(eig_num)
