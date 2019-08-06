from utilits import partitioning_qubo_build
import dimod
import numpy as np
from Koritskiy_Markowitz import Portfolio
from cplex_solver import CplexSolver


def permutation_exact_solver(mixed_matrix, theta=10):
    g_dim = mixed_matrix.shape[0]
    # Create binary quadratic model, that will find best permutation to return out mixed_matrix block structure
    qubo_matrix = partitioning_qubo_build(mixed_matrix, theta=theta)
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix, offset=2*g_dim)

    # Get solution by brute force
    response = dimod.ExactSolver().sample(bqm)
    solution = [int((i+1)/2) for i in response.first[0].values()]
    solution_permutation_matrix = np.asarray(solution).reshape((g_dim, g_dim))
    print("Solution permutation matrix")
    print(solution_permutation_matrix)
    print("\n")

    return solution_permutation_matrix


def dwave_solver(tasks: list, theta=None, num_reads=100):
    solution = []
    if theta is None:
        theta = [0.5, 0.5, 0]
    for task in tasks:
        prices, averages, covariance = task
        portfolio = Portfolio(theta=theta,
                              prices=prices,
                              averages=averages,
                              covariance=covariance)
        portfolio.to_qubo()
        # solution.append(portfolio.bruteforce())
        solution.append(portfolio.dwave(num_reads=num_reads))

    return solution


def cplex_solver(mixed_matrix, theta=10):
    dim = mixed_matrix.shape[0]
    # Create binary quadratic model, that will find best permutation to return out mixed_matrix block structure
    qubo_matrix = partitioning_qubo_build(mixed_matrix, theta=theta)
    cplexsolver = CplexSolver(qubo_matrix=qubo_matrix,
                              url="https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/",
                              api_key='api_642ddfbf-ae49-4947-84f1-196f6883eab2')
    qubo_solution = cplexsolver.find_opt()
    solution_permutation_matrix = np.array(qubo_solution).reshape((dim, dim))

    return solution_permutation_matrix
