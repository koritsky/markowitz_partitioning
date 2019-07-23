import numpy as np
import itertools


def qubo_build (matrix):
    n = matrix.shape[0]
    # build kind of similarity tensor
    iterations = itertools.product([i for i in range(n)], repeat=4)
    tensor = np.zeros((n, n, n, n))
    for it in iterations:
        i, j, k, l = it
        tensor[it] = abs(matrix[i, k])*((j-l)**2)
    reshaped_tensor = tensor.reshape(n**2, n**2)
    return reshaped_tensor


mat = np.eye(2,2)
print(qubo_build(mat))