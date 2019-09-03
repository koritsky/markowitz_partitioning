import numpy as np

from trash.partitioning import Partitioning

np.random.seed(4)
# Make console print look better
np.set_printoptions(precision=4,  # Digits after point
                    linewidth=170,  # Length of the line
                    suppress=True)  # Always fixed point notation

block_dim = [5, 1]
size = sum(block_dim)


block_mat = Partitioning.rand_sym_block_gen(block_dim)
# noise_mat = (2 * np.random.normal(0, 1, (size, size)) - 1)/2
noise_mat = np.zeros((size, size))
_, mixed_mat = Partitioning.mixed_matrix_generator(block_mat=block_mat)
mixed_mat = mixed_mat + noise_mat
part = Partitioning(mixed_mat)
_, solution = part.ga_solver()
part.permutation_mat = part.list_to_mat(solution)
permuted_mixed_mat = part.permute(part.permutation_mat, mixed_mat)
permuted_noise_mat = part.permute(part.permutation_mat, noise_mat)
new_block_mat = permuted_mixed_mat - permuted_noise_mat
print(new_block_mat)

