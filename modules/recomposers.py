from modules.markowitz import Portfolio
import numpy as np


class Recomposers:

    @staticmethod
    def permutation_decomposer(portfolio, permutation_mat, max_dim: int, theta=None):
        if theta is None:
            theta = portfolio.theta
        mixed_mat = portfolio.covariance
        averages = portfolio.averages
        prices = portfolio.prices
        ordered_mat = np.dot(permutation_mat.T,
                             np.dot(mixed_mat, permutation_mat))

        decomposed_matrices = Recomposers.split_until_threshold(ordered_mat, split_threshold=max_dim)
        ordered_averages = np.dot(averages, permutation_mat)
        ordered_prices = np.dot(prices, permutation_mat)
        dims = [mat.shape[0] for mat in decomposed_matrices]
        cuts = [sum(dims[:i]) for i in range(len(dims) + 1)]
        decomposed_averages = np.array([ordered_averages[cuts[i]: cuts[i + 1]] for i in range(len(cuts) - 1)])
        decomposed_prices = np.array([ordered_prices[cuts[i]: cuts[i + 1]] for i in range(len(cuts) - 1)])
        problems = list(zip(decomposed_prices, decomposed_averages, decomposed_matrices))
        portfolios = []
        for problem in problems:
            portfolio = Portfolio(prices=problem[0],
                                  averages=problem[1],
                                  covariance=problem[2],
                                  theta=theta)
            portfolios.append(portfolio)

        return portfolios


    @staticmethod
    def permutation_solution_composer(solutions, permutation_mat):
        # Get a solution for permutated matrix by concatenation solutions of subproblems
        solution = np.concatenate(solutions)

        # Get a solution of original matrix by applying inverse permutation matrix:
        original_solution = np.dot(solution, permutation_mat.T)

        return original_solution

    @staticmethod
    def split_metric(mat, split):
        if split is None:
            split = 60  # Because D-wave can solve up to ~64 node complete graph

        # By given split (number of column after which to cut mat) calculates metric -
        # average value of non diagonal block elements.

        dim = mat.shape[0]
        m = split + 1
        assert dim > m, "Split size more or equal to mat size"
        metric = 0
        for row in range(m, dim):
            for col in range(m):
                metric += abs(mat[row, col])
                return metric / ((dim - m) * m)
            pass

    @staticmethod
    def best_split(mat):
        # Calculates best split, that minimizes split_metric

        dim = mat.shape[0]
        metric, split = 10 ** 3, 0
        for spl in range(dim - 1):
            metr = Recomposers.split_metric(mat, spl)
            if metr < metric:
                metric, split = metr, spl
        return split

    @staticmethod
    def split_until_threshold(mat, split_threshold=None):
        # Split mat iteratively by 2 until each block has dimension less then threshold.

        if split_threshold is None:
            split_threshold = 60  # Because D-wave can solve up to ~64 node complete graph

        mat_to_split = [mat]
        good_mat = []
        while len(mat_to_split) != 0:
            temp_mat_to_split = []
            for i in range(len(mat_to_split)):
                mat = mat_to_split.pop(i)
                split = Recomposers.best_split(mat) + 1
                up_mat = mat[:split, :split]
                good_mat.append(up_mat) if split < split_threshold else temp_mat_to_split.append(up_mat)
                lo_mat = mat[split:, split:]
                good_mat.append(lo_mat) if (mat.shape[0] - split) < split_threshold else temp_mat_to_split.append(
                    lo_mat)
            mat_to_split.extend(temp_mat_to_split)
        return good_mat
