import numpy as np

from trash.partitioning import Partitioning
from markowitz import Portfolio
from trash.recomposers import Recomposers


def main():
    # Make console print look better
    np.set_printoptions(precision=2,  # Digits after point
                        linewidth=170,  # Length of the line
                        suppress=True)  # Always fixed point notation

    # %%

    # Fix seed to get non-random result
    # (Remove after debugging)
    np.random.seed(9)

    # Block dimensions to generate
    block_dim = [3, 1]
    size = sum(block_dim)

    # Generate random inputs for portfolio
    averages = np.random.rand(size)
    prices = np.random.rand(size)
    theta = [0.5, 0.5, 0]

    block_mat = Partitioning.rand_sym_block_gen(block_dim)
    _, mixed_mat = Partitioning.mixed_matrix_generator(block_mat=block_mat)
    covariance = mixed_mat

    # It also has to have covariance matrix. We assume that it is quazi-block.
    # We want to find blocks. With out toy example we would create block matrix (with block_dim)
    # Then we mix it and try to find proper permutation to return it's block shape.

    # A whole problem of finding permutation requires new class object.
    part = Partitioning(mixed_mat)  # It was given block_dim, so block matrix structure was generated automatically

    # Create main object: portfolio instance:
    portfolio = Portfolio(theta=theta,
                          averages=averages,
                          prices=prices,
                          covariance=covariance)

    # %%

    # Get solution_permutation matrix by exact solver
    _, solution = part.exact_solver()
    part.permutation_mat = part.list_to_mat(solution)
    print("Solution permutation matrix:")
    print(part.permutation_mat)
    print("\n")


    # Check, whether solution matrix is permutation one
    if not part.permutation_check(part.permutation_mat):
        print('\033[93m' + "Solution is not permutation matrix" +
              "Trying to fix it" + '\033[0m')
        new_solution_permutation_mat = part.to_permutation(part.permutation_mat)
        if part.permutation_check(new_solution_permutation_mat):
            print('\033[93m' + "Success!" + '\033[0m')
            part.permutation_mat = new_solution_permutation_mat
        print("Solution matrix after fixing:")
        print(part.permutation_mat)

    part.ordered_mat = part.permute(part.permutation_mat, part.mixed_mat)
    print("Ordered matrix:")
    print(part.ordered_mat)
    print("\n")
    # %%

    # Get a list of smaller portfolios which has covariance, prices and averages according to permutation
    portfolios = Recomposers.permutation_decomposer(portfolio=portfolio,
                                                    permutation_mat=part.permutation_mat,
                                                    max_dim=(max(block_dim) + 1))

    # %%
    # Here we get  several solutions

    # Solve each small portfolio task.
    solutions = []
    for port in portfolios:
        _, small_solution = port.dwave_solver(num_reads=1)
        solutions.append(small_solution)

    # Apply permutation to concatenated solution to get the solution of the original portfolio
    composed_solution = Recomposers.permutation_solution_composer(solutions, part.permutation_mat)
    print("Solution with partitioning:")
    print(portfolio.solution_energy(composed_solution), composed_solution)
    print("\n")


    ## Afterward genetic algorithm application ##
    # TODO: сделать топ n решений из dwave и засовывать их в ga как init_solution.
    # You can give it any iterable of good solutions, where ga would start from.

    new_prices = np.dot(prices, part.permutation_mat)
    new_averages = np.dot(averages, part.permutation_mat)

    new_portfolio = Portfolio(prices=new_prices,
                              averages=new_averages,
                              covariance=part.ordered_mat,
                              theta=theta)

    ga_energy, ga_solution = new_portfolio.ga_solver(iteration_number=10,
                                                     population_size=100,
                                                     init_solution=[composed_solution])
    ga_solution = np.dot(ga_solution,
                         part.permutation_mat.T)  # part.permutation_mat.T is inversed to part.permutation_mat

    print("GA solution:")
    print(ga_energy, ga_solution)

    # %%

    # Now compare it with exact solution of the whole matrix
    print("Now compare it with exact solution of the whole matrix")
    print(portfolio.exact_solver())
if __name__ == "__main__":
    main()
