import utilits
import solver
import decomposer
import composer
from Koritskiy_Markowitz import Portfolio
import numpy as np

# Choose number of digits after comma in console printout
after_comma = 3
block_dim = [1, 3]
np.random.seed(6)

# Generate random covariance matrix (here is called "mixed_matrix"), prices and averages
mixed_matrix = utilits.mixed_matrix_generator(block_dim)
prices = np.random.rand(sum(block_dim))
averages = np.random.rand(sum(block_dim))

# Get solution_permutation matrix
solution_permutation_matrix = solver.permutation_exact_solver(mixed_matrix)

bqm = utilits.partitioning_qubo_build(mixed_matrix)

# Check, whether solution matrix is permutation one
if not utilits.permutation_check(solution_permutation_matrix):
    print('\033[93m' + "Solution is not permutation matrix" + '\033[0m')
print(solution_permutation_matrix)

# Get decomposed matrices as a list of tuples (price, averages, covariance)
decomposed_matrices = decomposer.permutation_decomposer(mixed_matrix=mixed_matrix,
                                                        averages=averages,
                                                        prices=prices,
                                                        permutation_matrix=solution_permutation_matrix,
                                                        max_dim=(max(block_dim)+1))
# Get a list of solutions for decomposed matrices
solutions = solver.dwave_solver(decomposed_matrices, num_reads=100)

# Get a solution for original matrix
original_solution = composer.solution_composer(solutions=solutions,
                                               permutation_matrix=solution_permutation_matrix)
print("Solution:")
print(original_solution)
print("---------------------------------")

# ---------------------------------

# Now compare it with exact solution
print("Now compare it with exact solution")
portfolio = Portfolio(theta=[0.5, 0.5, 0],
                      covariance=mixed_matrix,
                      prices=prices,
                      averages=averages)
print("Exact solution:")
print(portfolio.bruteforce()[1])
