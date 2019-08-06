from __future__ import print_function
import random
import numpy as np
from pyeasyga import pyeasyga


import networkx as nx
import minorminer

from dwave_qbsolv import QBSolv
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
import dimod


class Portfolio:
    def __init__(self, budget=0,
                 theta=None,
                 prices=None,
                 averages=None,
                 covariance=None):
        if covariance is None:
            covariance = []
        if averages is None:
            averages = []
        if prices is None:
            prices = []
        if theta is None:
            theta = [0.3, 0.3, 0.3]
        self.prices = prices
        self.averages = averages
        self.covariance = covariance
        self.budget = budget
        self.theta = theta
        self.size = len(self.averages)
        self.qubo = np.zeros((self.size, self.size))
        self.ising = ()

    def file_read(self, file_prices, file_averages, file_covariance, threshold=0):
        f = open(file_prices, 'r')
        prices = [float(x) for x in f.readline().split(',')]
        self.size = len(prices) if threshold == 0 else threshold
        prices = np.asarray(prices)[:self.size]
        f.close()

        f = open(file_averages, 'r')
        averages = np.array([float(x) for x in f.readline().split(',')])[:self.size]
        f.close()

        covariance = np.zeros((self.size, self.size))
        f = open(file_covariance, 'r')
        i = 0
        for line in f:
            if i < self.size:
                values = line.split(',')
                for j in range(self.size):
                    covariance[i, j] = float(values[j])
            else:
                break
            i += 1
        f.close()

        self.prices, self.averages, self.covariance = prices, averages, covariance

    def to_ising(self):
        n = self.size
        qmatrix = np.zeros((n, n))
        qvector = np.zeros(n)

        for row in range(n):
            for col in range(row + 1, n, 1):
                qmatrix[row][col] = 2.0 * (self.theta[1] * self.covariance[row][col]
                                           + self.theta[2] * self.prices[row] * self.prices[col])

        for row in range(n):
            qvector[row] = -1.0 * self.theta[0] * self.averages[row] - \
                           self.theta[2] * 2.0 * self.budget * self.prices[row] + \
                           self.theta[1] * self.covariance[row][row] + self.theta[2] * self.prices[row] * self.prices[row]

        cfactor = self.theta[2] * self.budget * self.budget

        hvector = np.zeros(n)
        Jmatrix = 0.25 * qmatrix

        linear_offset = 0.0
        quadratic_offset = 0.0

        for i in range(n):
            hvector[i] = 0.5 * qvector[i]
            linear_offset += qvector[i]

        for row in range(n):
            for col in range( row + 1, n, 1 ):
                hvector[row] += 0.25 * qmatrix[row][col]
                hvector[col] += 0.25 * qmatrix[row][col]
                quadratic_offset += qmatrix[row][col]

        gfactor = cfactor + 0.5 * linear_offset + 0.25 * quadratic_offset
        h = dict(zip(range(len(hvector)), hvector))
        J = {}
        for i in range(n):
            for j in range(i+1, n):
                J[(i, j)] = Jmatrix[(i, j)]
        self.ising = (J, h, gfactor)
        return J, h, gfactor

    def to_qubo(self):
        for row in range(self.size):
            self.qubo[row, row] = (self.theta[1] * self.covariance[row][row]
                                   + self.theta[2] * self.prices[row] ** 2
                                   - self.theta[0] * self.averages[row]
                                   - 2 * self.budget * self.theta[2] * self.prices[row])
            for col in range(row + 1, self.size):
                self.qubo[row, col] = 2 * (self.theta[1] * self.covariance[row][col]
                                           + self.theta[2] * self.prices[row] * self.prices[col])
        return self.qubo

    def bruteforce(self):
        # TODO: To replace on dimod.ExactSolver()

        self.to_qubo()
        best_result = [None, []]
        for decimal in range(2 ** self.size):
            individual = str.zfill(bin(decimal).replace("0b", ""), self.size)
            individual = [int(i) for i in individual]
            result = \
                - self.theta[0] * np.dot(individual, self.averages) \
                + self.theta[1] * np.dot(individual, np.dot(self.covariance, individual)) \
                + self.theta[2] * (np.dot(individual, self.prices) - self.budget) ** 2
            if best_result[0] is None or result < best_result[0]:
                best_result = result, individual
        return best_result

    def genetic_algorithm(self, iteration_number=5, population_size=100, generations=50):
        data = self.prices, self.averages, self.covariance, self.theta, self.budget
        ga = pyeasyga.GeneticAlgorithm(data,
                                       population_size=population_size,
                                       generations=generations,
                                       crossover_probability=0.8,
                                       mutation_probability=0.05,
                                       maximise_fitness=False,
                                       elitism=True)

        def create_individual(data):
            return [random.randint(0, 1) for _ in range(len(data[0]))]

        def fitness(individual, data):
            individual = np.asarray(individual)
            [price, expectation, covariance, theta, budget] = data
            fit = \
                - theta[0] * np.dot(individual, expectation) \
                + theta[1] * np.dot(individual, np.dot(covariance, individual)) \
                + theta[2] * (np.dot(individual, price) - budget) ** 2
            return fit

        ga.fitness_function = fitness
        ga.create_individual = create_individual
        best_result = [None, []]
        for i in range(iteration_number):
            ga.run()
            if best_result[0] is None or ga.best_individual()[0] < best_result[0]:
                best_result = ga.best_individual()
        return best_result

    def qbsolv(self, fraction=0.1, num_repeats=50, quantum_solver=False):
        subgraph_size = int(self.size * fraction)
        # find embedding of subproblem-sized complete graph to the QPU
        if quantum_solver is True:
            G = nx.complete_graph(subgraph_size)
            system = DWaveSampler()
            embedding = minorminer.find_embedding(G.edges, system.edgelist)
            solver = FixedEmbeddingComposite(system, embedding)
        else:
            solver = 'tabu'

        # solve a random problem
        response = QBSolv().sample_qubo(self.Q,
                                        solver=solver,
                                        solver_limit=subgraph_size,
                                        num_repeats=num_repeats,
                                        timeout=30)
        best_result = (response.data_vectors['energy'], [])
        for i in range(self.size):
            best_result[1].append(int((response.samples()[0][i] + 1) / 2))
        print(response.data_vectors)
        return best_result

    def dwave(self, num_reads: int):
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import EmbeddingComposite
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(self.qubo)
        response = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=num_reads)
        solution = [response.samples()[0][i] for i in range(self.size)]
        return solution

    def results(self, solution: list):
        solution = np.asarray(solution)
        portfolio_return = np.dot(solution, self.averages)
        portfolio_variance = np.dot(solution, np.dot(self.covariance, solution))
        return portfolio_return, portfolio_variance


# if __name__ == "__main__":
#     import dimod
#
#     body = ['prices.csv', 'averages.csv', 'covariance.csv']
#     prefixes = ['up_', 'low_', 'merge_']
#
#     for prefix in prefixes:
#         portfolio = Portfolio( budget=100, theta=[0.5, 0.5, 0])
#         portfolio.file_read(prefix+body[0],
#                             prefix + body[1],
#                             prefix + body[2])
#         J, h, _ = portfolio.to_ising()
#         import time
#
#         start_time = time.time()
#         bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
#         exact_response = dimod.ExactSolver().sample(bqm)
#         exact_sollution = [int((i + 1) / 2) for i in exact_response.first[0].values()]
#         print( "--- %s seconds ---" % (time.time() - start_time) )
#         print(exact_sollution)
#
#
#
#     import sys
#     import hybrid
#
#     # load a problem
#     portfolio = Portfolio( budget=100, theta=[0.5, 0.5, 0] )
#     portfolio.file_read( prefixes[2] + body[0],
#                          prefixes[2] + body[1],
#                          prefixes[2] + body[2] )
#     J, h, _ = portfolio.to_ising()
#     bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
#
#     # run Tabu in parallel with QPU, but post-process QPU samples with very short Tabu
#     iteration = hybrid.Race(
#         hybrid.Identity(),
#         # hybrid.InterruptableTabuSampler(),
#         hybrid.EnergyImpactDecomposer(size=16)
#         | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=20)
#         | hybrid.SplatComposer()
#     ) | hybrid.ArgMin()
#
#     main = hybrid.Loop(iteration, max_iter=30, convergence=3)
#
#     # run the workflow
#     init_state = hybrid.State.from_sample( hybrid.utils.min_sample( bqm ), bqm )
#     solution = main.run( init_state ).result()
#
#     # show results
#     print([int((i + 1) / 2) for i in solution.samples.first[0].values()])