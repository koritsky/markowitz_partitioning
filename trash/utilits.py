import numpy as np
import itertools


def to_partitioning_qubo(cov_matrix, theta=10):
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
    while len(mat_to_split) != 0:
        temp_mat_to_split = []
        for i in range(len(mat_to_split)):
            mat = mat_to_split.pop(i)
            split = best_split(mat) + 1
            up_mat = mat[:split, :split]
            good_mat.append(up_mat) if split < threshold else temp_mat_to_split.append(up_mat)
            lo_mat = mat[split:, split:]
            good_mat.append(lo_mat) if (mat.shape[0]-split) < threshold else temp_mat_to_split.append(lo_mat)
        mat_to_split.extend(temp_mat_to_split)
    return good_mat


def permutation_check(matrix):
    mat = np.array(matrix)
    # Check, wheter given matrix is permutation one

    return np.array_equal(np.dot(mat, mat.T), np.eye(mat.shape[0]))


def mixed_matrix_generator(block_dim: list):
    # Generate random symmetric block matrix

    block_dim = block_dim
    matrix = np.around(rand_sym_block_gen(block_dim), 2)
    g_dim = sum(block_dim)
    print("Given random block matrix:")
    print(matrix)
    print("\n")

    # We define here (according to the letter) permutation matrix p_matrix as one
    # that have 1 on (i, j) if Permutation(i) = j.
    # So for column permutation of matrix A we should multiply  A*P. For rows (P.T)*A.
    # Hence (P.T)*A*P is matrix A with permuted columns and rows.

    # Generate random permutation matrix
    p_matrix = np.random.permutation(np.eye(g_dim))
    print("Random permutation matrix:")
    print(p_matrix)
    print("\n")

    # Mix generated matrix via permutation.
    mixed_matrix = np.dot(p_matrix.T, np.dot(matrix, p_matrix))
    print("Mixed matrix")
    print(mixed_matrix)
    print("\n")
    return mixed_matrix


def to_permutation(permutation_matrix, bqm):
    mat = np.array(permutation_matrix)
    dim = mat.shape[0]

    row_holes = []
    col_holes = []

    for i in range(dim):
        if sum(mat[i]) != 1:
            row_holes.append(i)
            mat[i] = np.zeros(dim)
        if sum(mat.T[i]) != 1:
            col_holes.append(i)
            mat.T[i] = np.zeros(dim)
    if len(row_holes) != len(col_holes):
        print('\033[93m' + "Unable to fix this shit ¯\_(ツ)_/¯" + '\033[0m')

    sub_dim = len(row_holes)
    print('\033[93m' + "Number of wrong lines:" + '\033[0m')
    print('\033[93m' + str(sub_dim) + '\033[0m')
    assert sub_dim < 9, "Too many mistakes to fix it by bruteforce"
    solutions_energy = []

    permutations = [np.array(p) for p in itertools.permutations(np.eye(sub_dim))]
    holes = list(itertools.product(row_holes, col_holes))
    simple_iters = list(itertools.product(range(sub_dim), repeat=2))
    mega_iters = list(zip(holes, simple_iters))
    for permutation in permutations:
        sub_mat = np.array(permutation)
        hole_mat = np.zeros((dim, dim))
        for iter in mega_iters:
            hole_mat[iter[0]] = sub_mat[iter[1]]
        new_mat = mat + hole_mat
        solutions_energy.append(bqm.energy(new_mat.reshape(dim ** 2)))

    permutation_index = solutions_energy.index(min(solutions_energy))

    sub_mat = permutations[permutation_index]
    hole_mat = np.zeros((dim, dim))
    for iter in mega_iters:
        hole_mat[iter[0]] = sub_mat[iter[1]]
    new_mat = mat + hole_mat
    return new_mat
