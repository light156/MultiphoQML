import torch
import numpy as np
import scipy as sp


def perm_recursive(matrix):
    def _permanent(mtx, column, selected, prod):
        """
        Row expansion for the permanent of matrix mtx.
        The counter column is the current column,
        selected is a list of indices of selected rows,
        and prod accumulates the current product.
        """
        if column == mtx.shape[1]:
            return prod
        else:
            result = 0
            for row in range(mtx.shape[0]):
                if not row in selected:
                    result += _permanent(mtx, column + 1, selected + [row], prod * mtx[row, column])
            return result

    return _permanent(matrix, 0, [], 1)


def perm(matrix):
    """
    Calculate the permanent of matrix A using bbfg method.
    Adapted from thewalrus package

    Returns:
        the permanent of matrix ``A``
    """
    A = torch.as_tensor(matrix)

    n = A.shape[0]

    if torch.isnan(A).any():
        raise ValueError("Input matrix must not contain NaNs.")

    if n == 0:
        return A.dtype.type(1.0)

    if n == 1:
        return A[0, 0]

    if n == 2:
        return A[0, 0] * A[1, 1] + A[0, 1] * A[1, 0]


    if n == 3:
        return (
            A[0, 2] * A[1, 1] * A[2, 0]
            + A[0, 1] * A[1, 2] * A[2, 0]
            + A[0, 2] * A[1, 0] * A[2, 1]
            + A[0, 0] * A[1, 2] * A[2, 1]
            + A[0, 1] * A[1, 0] * A[2, 2]
            + A[0, 0] * A[1, 1] * A[2, 2]
        )

    if n == 4:
        return (
            A[0, 0] * A[1, 1] * A[2, 2] * A[3, 3]
            + A[0, 0] * A[1, 1] * A[2, 3] * A[3, 2]
            + A[0, 0] * A[1, 2] * A[2, 1] * A[3, 3]
            + A[0, 0] * A[1, 2] * A[2, 3] * A[3, 1]
            + A[0, 0] * A[1, 3] * A[2, 1] * A[3, 2]
            + A[0, 0] * A[1, 3] * A[2, 2] * A[3, 1]

            + A[0, 1] * A[1, 0] * A[2, 2] * A[3, 3]
            + A[0, 1] * A[1, 0] * A[2, 3] * A[3, 2]
            + A[0, 1] * A[1, 2] * A[2, 0] * A[3, 3]
            + A[0, 1] * A[1, 2] * A[2, 3] * A[3, 0]
            + A[0, 1] * A[1, 3] * A[2, 0] * A[3, 2]
            + A[0, 1] * A[1, 3] * A[2, 2] * A[3, 0]

            + A[0, 2] * A[1, 0] * A[2, 1] * A[3, 3]
            + A[0, 2] * A[1, 0] * A[2, 3] * A[3, 1]
            + A[0, 2] * A[1, 1] * A[2, 0] * A[3, 3]
            + A[0, 2] * A[1, 1] * A[2, 3] * A[3, 0]
            + A[0, 2] * A[1, 3] * A[2, 0] * A[3, 1]
            + A[0, 2] * A[1, 3] * A[2, 1] * A[3, 0]

            + A[0, 3] * A[1, 0] * A[2, 1] * A[3, 2]
            + A[0, 3] * A[1, 0] * A[2, 2] * A[3, 1]
            + A[0, 3] * A[1, 1] * A[2, 0] * A[3, 2]
            + A[0, 3] * A[1, 1] * A[2, 2] * A[3, 0]
            + A[0, 3] * A[1, 2] * A[2, 0] * A[3, 1]
            + A[0, 3] * A[1, 2] * A[2, 1] * A[3, 0]
        )
    
    # row_comb keeps the sum of previous subsets.
    # Every iteration, it removes a term and/or adds a new term
    # to give the term to add for the next subset
    row_comb = torch.zeros(n, dtype=A.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = [2 ** i for i in range(n)]
    num_loops = 2 ** n
    for k in range(0, num_loops):
        bin_index = (k + 1) % num_loops
        reduced = torch.prod(row_comb)
        total += sign * reduced
        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_dict.index(grey_diff)
        new_vector = A[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)
        for i in range(n):
            row_comb[i] += new_vector[i] * direction
        sign = -sign
        old_grey = new_grey
    return total


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


def gen_random_unitary(N, real=False):
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
    import time

    time1, time2 = 0, 0

    for _ in range(5):
        unitary = torch.as_tensor(gen_random_unitary(6))
        
        start = time.time()
        res1 = perm(unitary)
        end = time.time()

        time1 += end-start

        start = time.time()
        res2 = perm_recursive(unitary)
        end = time.time()

        time2 += end-start

        print(res1-res2)
    
    print(time1, time2)

