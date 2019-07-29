import numpy as np
import itertools


def qubo_build(cov_matrix, theta=1):
    # Dimension of primary task
    n = cov_matrix.shape[0]

    # Tensor that goes into objective function, 4 variables
    measure_tensor = np.zeros((n, n, n, n))

    # Matrix of qubo form
    qubo_matrix = np.zeros((n ** 2, n ** 2))

    # Reshape tensor into qubo matrix
    for it in itertools.product([i for i in range(n)], repeat=4):
        i, j, k, l = it
        measure_tensor[i, j, k, l] = abs(cov_matrix[i, j]) * ((k-l) ** 2)
        qubo_matrix[i * n + k][j * n + l] = measure_tensor[i, j, k, l]

    # Add constraints
    qubo_matrix = qubo_matrix + theta * ((-4)*np.eye(n ** 2)
                                         + np.kron(np.eye(n), np.ones((n, n)))
                                         + np.kron(np.ones((n, n)), np.eye(n)))
    # Make it upper-diagonal
    qubo_matrix = qubo_matrix + np.triu(qubo_matrix, 1) - np.tril(qubo_matrix, -1)
    return qubo_matrix


def rand_sym_block_gen(block_dim: list):
    # Check for valid block_dim list
    for i in block_dim:
        assert i > 0, "Nonpositive block dimension"

    general_dim = sum(block_dim)  # Dimension of resulting matrix
    matrix = np.zeros((general_dim, general_dim))

    current_dim = 0
    for dim in block_dim:
        current_dim = current_dim + dim
        block = np.random.rand(dim, dim)
        matrix += matrix + np.pad(block,
                                  ((current_dim - dim, general_dim - current_dim),
                                   (current_dim - dim, general_dim - current_dim)),
                                  "constant",
                                  constant_values=(0, 0))
    # Make it symmetric
    matrix = (1 / 2) * (matrix + matrix.T)
    return matrix


def split_metric(matrix, split: int):
    # By given split (after which column cut matrix) calculates metric - average value of non diagonal block elements.

    n = matrix.shape[0]
    m = split + 1
    assert n > m, "Split size more or equal to matrix size"
    metric = 0
    for row in range(m, n):
        for col in range(m):
            metric += abs(matrix[row, col])
    return metric / ((n - m) * m)


def best_split(matrix):
    # Calculates best split, that minimizes split_metric

    dim = matrix.shape[0]
    metric, split = 10 ** 3, 0
    for spl in range(dim - 1):
        metr = split_metric(matrix, spl)
        if metr < metric:
            metric, split = metr, spl
    return split


def split_until_threshold(matrix, threshold):
    # Split matrix iteratively by 2 until each block has dimension less then threshold.

    mat_to_split = [matrix]
    good_mat = []
    while len(mat_to_split) !=0:
        temp_mat_to_split = []
        for i in range(len(mat_to_split)):
            mat = mat_to_split.pop(i)
            split = best_split(mat) + 1
            up_mat = mat[:split, :split]
            good_mat.append(up_mat) if split < threshold else temp_mat_to_split.append(up_mat)
            lo_mat = mat[split :, split:]
            good_mat.append(lo_mat) if (mat.shape[0]-split) < threshold else temp_mat_to_split.append(lo_mat)
        mat_to_split.extend(temp_mat_to_split)
    return good_mat


def permutation_check(matrix):
    #Check, wheter given matrix is permutation one

    return np.array_equal(np.dot(matrix, matrix.T), np.eye(matrix.shape[0]))




