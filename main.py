import utilits
import solver
import decomposer
import composer
import dimod
from Koritskiy_Markowitz import Portfolio
import numpy as np

# Make console print look better
np.set_printoptions(precision=2,  # Digits after point
                    linewidth=170,  # Length of the line
                    suppress=True)  # Always fixed point notation

# Choose number of digits after comma in console printout
after_comma = 3
block_dim = [5, 5, 5, 5, 5]
np.random.seed(6)

# Generate random covariance matrix (here is called "mixed_matrix"), prices and averages
mixed_matrix = utilits.mixed_matrix_generator(block_dim)
prices = np.random.rand(sum(block_dim))
averages = np.random.rand(sum(block_dim))

# Get solution_permutation matrix
solution_permutation_matrix = solver.cplex_solver(mixed_matrix)

# Check, whether solution matrix is permutation one
if not utilits.permutation_check(solution_permutation_matrix):
    print('\033[93m' + "Solution is not permutation matrix" + '\033[0m')
    print('\033[93m' + "I'll try to fix it " + '\033[0m')
    qubo_matrix = utilits.partitioning_qubo_build(mixed_matrix)
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
    new_solution_permutation_matrix = utilits.to_permutation(solution_permutation_matrix, bqm)
    if utilits.permutation_check(new_solution_permutation_matrix):
        print('\033[93m' + "Success!" + '\033[0m')
        print('\033[93m' + "New matrix is permutation" + '\033[0m')
        solution_permutation_matrix = new_solution_permutation_matrix
    print("Solution matrix after fixing:")
    print(solution_permutation_matrix)

# Get decomposed matrices as a list of tuples (price, averages, covariance)
decomposed_matrices, ordered_matrix = decomposer.permutation_decomposer(mixed_matrix=mixed_matrix,
                                                                        averages=averages,
                                                                        prices=prices,
                                                                        permutation_matrix=solution_permutation_matrix,
                                                                        max_dim=(max(block_dim) + 1))
# Get a list of solutions for decomposed matrices
solutions = solver.dwave_solver(decomposed_matrices, num_reads=10)

# Get a solution for original matrix
original_solution = composer.solution_composer(solutions=solutions,
                                               permutation_matrix=solution_permutation_matrix)

# Genetic algorithm application:
portfolio = Portfolio(prices=prices,
                      averages=averages,
                      covariance=ordered_matrix,
                      theta=[0.5, 0.5, 0])
# TODO: сделать топ n решений из dwave и засовывать их в ga как init_solution.
# You can give it any iterable of good solutions, where ga would start from.
ga_solution = portfolio.genetic_algorithm(init_solution=[original_solution], iteration_number=100)

print("Original solution:")
print(original_solution)
print("GA solution:")
print(ga_solution)
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
