import numpy as np
import dimod
import time
from markowitz import Markowitz
from cimsim.cimsim import CIMSim
import partitioning_utilits as pu
from to_ising_file import to_ising_file, ising_to_matrix

# How many digits to print in console
after_comma = 2

# Generate random symmetric block matrix
matrix = [[1.1, 1.36, 1.39, 0.63, 1.4, 0., 0., 0., 0., 0.],
          [1.36, 0.88, 1.42, 0.98, 1.18, 0., 0., 0., 0., 0.],
          [1.39, 1.42, 1.14, 1.76, 0.53, 0., 0., 0., 0., 0.],
          [0.63, 0.98, 1.76, 1.56, 1.65, 0., 0., 0., 0., 0.],
          [1.4, 1.18, 0.53, 1.65, 0.24, 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0.64, 0.2, 0.78, 0.44, 0.54],
          [0., 0., 0., 0., 0., 0.2, 0.77, 0.53, 0.5, 0.11],
          [0., 0., 0., 0., 0., 0.78, 0.53, 0.62, 0.82, 0.41],
          [0., 0., 0., 0., 0., 0.44, 0.5, 0.82, 0.06, 0.49],
          [0., 0., 0., 0., 0., 0.54, 0.11, 0.41, 0.49, 0.36]]

# np.random.seed(0)
# block_dim = [3, 1]
# matrix = pu.rand_sym_block_gen(block_dim)
# g_dim = sum(block_dim)
g_dim = 10

print("Given random block matrix:")
print(np.around(matrix, after_comma))
print("\n")

"""
We define here (according to the letter) permutation matrix p_matrix as one
that have 1 on (i, j) if Permutation(i) = j on columns.
So for column permutation of matrix A we should multiply  AP. For rows (P.T)A.
Hence (P.T)AP is matrix A with permuted columns and rows.
"""

# Generate random permutation matrix
p_matrix = np.random.permutation(np.eye(g_dim))
print("Random permutation matrix:")
print(p_matrix)
print("\n")

# Mix generated matrix via permutation.
mixed_matrix = np.dot(p_matrix.T, np.dot(matrix, p_matrix))
print("Mixed matrix")
print(np.around(mixed_matrix, after_comma))
print("\n")

# Create binary quadratic model, that will find best permutation and return block structure of mixed_matrix
qubo_matrix = pu.qubo_build(mixed_matrix, theta=70)
bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)

# Transform it to ising matrix and vector
h, J, _ = bqm.to_ising()
bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
hvector, Jmatrix = ising_to_matrix(h, J)

# Generate "ising file" for ParameterOptimizer
to_ising_file(hvector, Jmatrix, "block_4_ising.txt")


# Run CIMSim
cimsim = CIMSim(Jmatrix, hvector, device="cpu")
cimsim.set_params({'c_th': 1.,
                   'zeta': 1.,
                   'init_coupling': 0.3,
                   'final_coupling': 1.,
                   'N': 1000,
                   'attempt_num': 30000,
                   'dt': 0.000084,
                   'sigma': 1758.101,
                   'alpha': 0.92087773,
                   'S': 0.02758990,
                   'D': -0.001519,
                   'O': 95.9835})
(spins_ising, energy_ising, c_current, c_evol) = cimsim.find_opt()

print(energy_ising)
# Print solution as permutation matrix
solution = [int((i + 1) / 2) for i in spins_ising]
solution_permutation_matrix = np.asarray(solution).reshape((g_dim, g_dim))
print("Solution permutation matrix")
print(solution_permutation_matrix)
print("\n")

# Check, whether solution matrix is permutation one.
if not pu.permutation_check(solution_permutation_matrix):
    print("Solution is not permutation matrix")

# Apply solution matrix to mixed and get the block one.
new_matrix = np.dot(solution_permutation_matrix.T,
                    np.dot(mixed_matrix,
                           solution_permutation_matrix))
print("New matrix")
print(np.around(new_matrix, after_comma))
print("\n")
