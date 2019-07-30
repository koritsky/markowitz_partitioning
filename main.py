import dimod
from utilits import *

# Choose number of digits after comma in console printout
after_comma = 3


# Generate random symmetric block matrix
block_dim = [3, 1]
matrix = rand_sym_block_gen(block_dim)
g_dim = sum(block_dim)
print("Given random block matrix:")
print(np.around(matrix, after_comma))
print("\n")


# We define here (according to the letter) permutation matrix p_matrix as one
# that have 1 on (i, j) if Permutation(i) = j.
# So for column permutation of matrix A we should multiply  AP. For rows (P.T)A.
# Hence (P.T)AP is matrix A with permuted columns and rows.


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

# Create binary quadratic model, that will find best permutation to return out mixed_matrix block structure
qubo_matrix = qubo_build(mixed_matrix, theta=10)
bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix, offset=2*g_dim)

# Get solution by brute force
response = dimod.ExactSolver().sample(bqm)
solution = [int((i+1)/2) for i in response.first[0].values()]
solution_permutation_matrix = np.asarray(solution).reshape((g_dim, g_dim))
print("Solution permutation matrix")
print(solution_permutation_matrix)
print("\n")

# Check, whether solution matrix is permutation one.
if not permutation_check(solution_permutation_matrix):
    print("Solution is not permutation matrix")

# Apply solution matrix to mixed and get the block one.
new_matrix = np.dot(solution_permutation_matrix.T,
                    np.dot(mixed_matrix,
                           solution_permutation_matrix))
print("New matrix")
print(np.around(new_matrix, after_comma))
print("\n")

# Split new matrix into blocks with block dimension less then threshold
print("Splitted matrices:")
for i in split_until_threshold(new_matrix, threshold=8):
    print(np.around(i, after_comma))








