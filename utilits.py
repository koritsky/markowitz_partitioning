import numpy as np
import itertools


def qubo_build(cov_matrix, theta=1):
    '''Create a qubo matrix by given covariance matrix

    :param cov_matrix: covariance matrix from the problem. Must be symmetric.
    :param theta: factor that determines how to penalize for violation of one-to-one permutation constrain.
    Must be >0.
    :return: QUBO matrix. Diagonal elements - linear coefficients. Non diagonal - quadratic.
    Resulting QUBO matrix is upper diagonal
    '''

    # dimension of primary task
    n = cov_matrix.shape[0]

    # tensor that goes into objective function, 4 variables
    measure_tensor = np.zeros((n, n, n, n))

    # matrix of qubo form
    qubo_matrix = np.zeros((n ** 2, n ** 2))

    # reshape tensor into qubo matrix
    for it in itertools.product([i for i in range(n)], repeat=4):
        i, j, k, l = it
        measure_tensor[i, j, k, l] = abs(cov_matrix[i, j]) * ((k-l) ** 2)
        qubo_matrix[i * n + k][j * n + l] = measure_tensor[i, j, k, l]

    # add constraints
    qubo_matrix = qubo_matrix + theta * ((-4)*np.eye(n ** 2)
                                         + np.kron(np.eye(n), np.ones((n, n)))
                                         + np.kron(np.ones((n, n)), np.eye(n)))
    # make it upper-diagonal
    # qubo_matrix = qubo_matrix + np.triu(qubo_matrix, 1) - np.tril(qubo_matrix, -1)
    return qubo_matrix


def rand_sym_block_gen(block_dim:list):
    '''Generates random symmetric block matrix
    :param general_dim: dimension of the whole matrix
    :param block_dim: dimension of  main block
    :return:
    '''
    for i in block_dim:
        assert i > 0, "Nonpositive block dimension"
    general_dim = sum(block_dim)
    matrix = np.zeros((general_dim, general_dim))
    current_dim = 0
    for dim in block_dim:
        current_dim = current_dim + dim
        block =  np.random.rand(dim, dim)
        matrix += matrix + np.pad(block,
                                  ((current_dim - dim, general_dim - current_dim),
                                   (current_dim - dim, general_dim - current_dim)),
                                  "constant",
                                  constant_values=(0, 0))
    matrix = (matrix + matrix.T) / 2
    return matrix

def split_metric(matrix, split:int):
    """Metric of matric splitting based on mean of non block elements
    :param matrix:
    :param split: index of last object in first part
    :return: int, closer to zero - better split.
    """
    n = matrix.shape[0]
    m = split + 1
    assert n > m, "Split size more or equal to matrix size"
    metric = 0
    for row in range(m, n):
        for col in range(m):
            metric += abs(matrix[row, col])
    return metric/((n - m) * m)


def best_split(matrix):
    dim = matrix.shape[0]
    metric, split = 10 ** 3, 0
    for spl in range(dim - 1):
        metr = split_metric(matrix, spl)
        if metr < metric:
            metric, split = metr, spl
    return split

def split_until_threshold(matrix, threshold):
    mat_to_split = []
    mat_to_split.append(matrix)
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




