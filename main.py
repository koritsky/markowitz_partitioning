import numpy as np
import dimod
from utilits import qubo_build, rand_sym_block_gen, split_until_threshold
from hybrid.reference.kerberos import KerberosSampler

around = 3

# generate block matrix
block_dim = [3, 5, 4]
g_dim = sum(block_dim)
matrix = rand_sym_block_gen(block_dim)
print("Given random block matrix:")
print(np.around(matrix, around))
print("\n")
"""
We define here (according to the letter) permutation matrix p_matrix as one 
that have 1 on (i, j) if Permutation(i) = j.
So for column permutation of matrix A we should multiply  AP. For rows (P.T)A.
Hence (P.T)AP is matrix A with permuted columns and rows.
"""


# generate permutation matrix
p_matrix = np.random.permutation(np.eye(g_dim))
print("Permutation matrix:")
print(p_matrix)
print("\n")

mixed_matrix = np.dot(p_matrix.T, np.dot(matrix, p_matrix))
print("Mixed matrix")
print(np.around(mixed_matrix, around))
print("\n")
qubo_matrix = qubo_build(mixed_matrix, theta=30)
bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix, offset=2*g_dim)
# response = dimod.ExactSolver().sample(bqm)
response = KerberosSampler().sample(bqm)
exact_solution = [int((i + 1) / 2) for i in response.first[0].values()]
solution_permutation_matrix = np.asarray(exact_solution).reshape((g_dim, g_dim))
print("Solution permutation matrix")
print(solution_permutation_matrix)
print("\n")

print("New matrix")
new_matrix = np.dot(solution_permutation_matrix.T, np.dot(mixed_matrix, solution_permutation_matrix))
print(np.around(new_matrix, around))
print("\n")
print("Splitted matrices:")
print("\n")
for i in split_until_threshold(new_matrix, threshold=6):
    print(np.around(i, around))




