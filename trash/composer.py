import numpy as np

def solution_composer(solutions, permutation_matrix):
    # Get a solution for permutated matrix by concatenation solutions of subproblems
    solution = np.concatenate(solutions)

    # Get a solution of original matrix by applying inverse permutation matrix:
    original_solution = np.dot(solution, permutation_matrix.T)

    return original_solution