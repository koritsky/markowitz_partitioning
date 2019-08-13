import numpy as np
import dimod

from binary_problem import BinaryProblem
from partitioning import Partitioning
from markowitz import Portfolio
import decomposer
import composer


def main():
    # Make console print look better
    np.set_printoptions(precision=2,  # Digits after point
                        linewidth=170,  # Length of the line
                        suppress=True)  # Always fixed point notation

    # Fix seed to get non-random result
    # (Remove after debugging)
    np.random.seed(5)


    # Block dimensions to generate
    block_dim = [3, 1]
    size = sum(block_dim)

    # Create main object: portfolio instance:
    portfolio = Portfolio(theta=[0.5, 0.5, 0])

    portfolio.averages = np.random.rand(size)
    portfolio.prices = np.random.rand(size)

    # It also has to have covariance matrix. We assume that it is quazi-block.
    # We want to find block. With out toy example we would create block matrix (with block_dim)
    # Then we mix it and try to find proper permutation to return it block shape.

    # A whole problem of finding permutation requires new class object.
    part = Partitioning(block_dim)
    # Get solution_permutation matrix
    _, solution = part.exact_solver()
    solution_permutation_mat = part.list_to_mat(solution)
    print(np.dot(solution_permutation_mat.T, np.dot(part.mixed_mat, solution_permutation_mat)))

    # Check, whether solution matrix is permutation one
    #TODO: 1. Move it to partitioning, 2. Made it to use ga if lots of errors
    if not part.permutation_check(solution_permutation_mat):
        print('\033[93m' + "Solution is not permutation matrix" + '\033[0m')
        # print('\033[93m' + "I'll try to fix it " + '\033[0m')
        # qubo_matrix = part.to_partitioning_qubo(p.mixed_mat)
        # bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        # new_solution_permutation_matrix = utilits.to_permutation(solution_permutation_matrix, bqm)
        # if utilits.permutation_check(new_solution_permutation_matrix):
        #     print('\033[93m' + "Success!" + '\033[0m')
        #     print('\033[93m' + "New matrix is permutation" + '\033[0m')
        #     solution_permutation_matrix = new_solution_permutation_matrix
        # print("Solution matrix after fixing:")
        # print(solution_permutation_matrix)

    # Get decomposed matrices as a list of tuples (price, averages, covariance)
    decomposed_matrices, ordered_matrix = decomposer.permutation_decomposer(mixed_matrix=part.mixed_mat,
                                                                            averages=portfolio.averages,
                                                                            prices=portfolio.prices,
                                                                            permutation_matrix=solution_permutation_mat,
                                                                            max_dim=(max(block_dim) + 1))
    # Get a list of solutions for decomposed matrices
    solutions = binary_problem.dwave_solver(decomposed_matrices, num_reads=10)

    # Get a solution for original matrix
    original_solution = composer.solution_composer(solutions=solutions,
                                                   permutation_matrix=solution_permutation_matrix)

    # # Genetic algorithm application:
    # portfolio = Portfolio(prices=p.prices,
    #                       averages=p.averages,
    #                       covariance=ordered_matrix,
    #                       theta=[0.5, 0.5, 0])
    # # TODO: сделать топ n решений из dwave и засовывать их в ga как init_solution.
    # # You can give it any iterable of good solutions, where ga would start from.
    # ga_solution = portfolio.genetic_algorithm(init_solution=[original_solution], iteration_number=100)
    #
    # print("Original solution:")
    # print(original_solution)
    # print("GA solution:")
    # print(ga_solution)
    # print("---------------------------------")

    # ---------------------------------

    # Now compare it with exact solution
    print("Now compare it with exact solution")
    portfolio = Portfolio(theta=[0.5, 0.5, 0],
                          covariance=p.mixed_mat,
                          prices=p.prices,
                          averages=p.averages)
    print("Exact solution:")
    print(portfolio.bruteforce()[1])


if __name__ == "__main__":
    main()
