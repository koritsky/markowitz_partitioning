from partitioning import Partitioning
from markowitz import Portfolio
import numpy as np

class Recomposers:
    @staticmethod
    def permutation_decomposer(mixed_matrix, averages, prices, permutation_matrix, max_dim: int, ):
        new_matrix = np.dot(permutation_matrix.T,
                            np.dot(mixed_matrix, permutation_matrix))

        print("New matrix")
        print(new_matrix)
        print("\n")

        decomposed_matrices = utilits.split_until_threshold(new_matrix, threshold=max_dim)

        averages = np.dot(averages, permutation_matrix)
        prices = np.dot(prices, permutation_matrix)
        dims = [mat.shape[0] for mat in decomposed_matrices]
        cuts = [sum(dims[:i]) for i in range(len(dims) + 1)]
        decomposed_averages = np.array([averages[cuts[i]: cuts[i + 1]] for i in range(len(cuts) - 1)])
        decomposed_prices = np.array([prices[cuts[i]: cuts[i + 1]] for i in range(len(cuts) - 1)])
        return list(zip(decomposed_prices, decomposed_averages, decomposed_matrices)), new_matrix

