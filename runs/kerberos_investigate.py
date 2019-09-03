from modules.partitioning import Partitioning
import numpy as np
from hybrid.reference.kerberos import KerberosSampler

np.set_printoptions(precision=2,  # Digits after point
                    linewidth=170,  # Length of the line
                    suppress=True)  # Always fixed point notation (not scientific)

block_dim = 334
size = 9


def path(name):
    return f"/home/koritskiy/rqc/Markowitz_partitioning/data/test_matrices/{block_dim}/{name}{block_dim}.csv"


block_mat = np.genfromtxt(path("block_mat"), delimiter=",")
mix_permutation_mat = np.genfromtxt(path("mix_permutation_mat"), delimiter=",")
mixed_mat = np.genfromtxt(path("mixed_mat"), delimiter=",")
part = Partitioning(mixed_mat, theta=500)

response = KerberosSampler().sample(part.bqm, max_subproblem_size=60)
solution = np.array([int(i) for i in response.first[0].values()])

ordered_mat = part.permute(part.list_to_mat(solution))
print(ordered_mat)




