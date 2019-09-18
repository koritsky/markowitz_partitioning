"""
We want just to change Kerberos: substitute dwave solver with simCIM solver. For it we wouldn't build new workflow.
Instead, we add new possibility to module kerberos.py.
"""




from modules.partitioning import Partitioning
from modules.cim_kerberos import KerberosSampler
import numpy as np
import hybrid

np.set_printoptions(precision=2,  # Digits after point
                    linewidth=170,  # Length of the line
                    suppress=True)  # Always fixed point notation (not scientific)

block_dim = 334


# define the file path
def path(name):
    return f"/home/koritskiy/rqc/Markowitz_partitioning/data/test_matrices/{block_dim}/{name}{block_dim}.csv"


# download matrices
block_mat = np.genfromtxt(path("block_mat"), delimiter=",")
mix_permutation_mat = np.genfromtxt(path("mix_permutation_mat"), delimiter=",")
mixed_mat = np.genfromtxt(path("mixed_mat"), delimiter=",")

# Create object of partitioning task
part = Partitioning(mixed_mat, theta=500)
bqm = part.bqm

# Solve it using cim_kerberos solver
response = KerberosSampler().sample(bqm,
                                    max_subproblem_size=60,
                                    num_reads=5,
                                    simcim=True)
solution = np.array([int(i) for i in response.first[0].values()])


# Apply solution permutation matrix
ordered_mat = part.permute(part.list_to_mat(solution))
print(ordered_mat)
