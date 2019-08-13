import dimod
import numpy as np
from hybrid.reference.kerberos import KerberosSampler
from cplex_solver import CplexSolver
from pyeasyga import pyeasyga_kor
import random


class BinaryProblem:
    # TODO: add assertions on empty bqm for all solvers:
    # def qubo_check(self):
    #     """ Check, whether qubo model was built.
    #     """
    #     if np.array_equal(self.portfolio_qubo, np.zeros((self.size, self.size))):
    #         self.to_qubo()

    def __init__(self, bqm=None, qubo_mat=None):
        if bqm is None:
            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        self.bqm = bqm
        self.size = 0
        if qubo_mat is None:
            qubo_mat = np.zeros((self.size, self.size))
        self.qubo_mat = qubo_mat
        self.current_solution = 0, []

    def from_qubo(self, qubo_mat):
        self.qubo_mat = np.array(qubo_mat)
        self.size = self.qubo_mat.shape[0]
        self.qubo_to_bqm()

    def qubo_to_bqm(self):
        self.bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(self.qubo_mat)

    def solution_energy(self, solution):
        return dimod.BinaryQuadraticModel.energy(self.bqm, solution)

    def exact_solver(self):
        # Get solution by brute force
        response = dimod.ExactSolver().sample(self.bqm)
        solution = [int((i+1)/2) for i in response.first[0].values()]
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution

    def dwave_solver(self, num_reads=100):
        """ Solves using Dwave solver
                :param num_reads:
                :return: tuple (energy, solution)
                """
        # TODO: add sapi_token, dwave_url checker
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import EmbeddingComposite
        response = EmbeddingComposite(DWaveSampler()).sample(self.bqm, num_reads=num_reads)
        solution = np.array([int(i) for i in response.first[0].values()])
        print(solution)
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution

    def cplex_solver(self):
        # Create binary quadratic model, that will find best permutation to return out mixed_matrix block structure
        cplexsolver = CplexSolver(qubo_matrix=self.qubo_mat,
                                  url="https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/",
                                  api_key='api_642ddfbf-ae49-4947-84f1-196f6883eab2')
        solution = cplexsolver.find_opt()
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution

    def kerberos_solver(self, max_iter=10, qpu_reads=100):
        response = KerberosSampler().sample(self.bqm,
                                            max_iter=max_iter,
                                            convergence=3,
                                            qpu_reads=qpu_reads)
        solution = np.array([int(i) for i in response.first[0].values()])
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution

    def genetic_algorithm(self,
                          iteration_number=5,
                          population_size=100,
                          generations=50,
                          init_solution=None,
                          create_individual_func=None,
                          ):
        """Solves using genetic algorithm
        :param iteration_number: Number of iteration of genetic algorithm
        :param population_size:
        :param generations:
        :param init_solution: list List of initial "good" solutions to start with
        :return: tuple (energy, solution)
        """

        data = self.bqm
        ga = pyeasyga_kor.GeneticAlgorithm(data,
                                           population_size=population_size,
                                           generations=generations,
                                           maximise_fitness=False,
                                           elitism=True)

        def create_individual(data):
            return [random.randint(0, 1) for _ in range(self.size)]
        if create_individual_func is None:
            create_individual_func = create_individual

        def fitness(individual, data):
            return dimod.BinaryQuadraticModel.energy(data, individual)

        ga.fitness_function = fitness
        ga.create_individual = create_individual_func
        ga.create_initial_population(initial_genes=init_solution)
        best_result = None, []
        for i in range(iteration_number):
            ga.run()
            if best_result[0] is None or ga.best_individual()[0] < best_result[0]:
                best_result = ga.best_individual()
        solution = best_result
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution


    # I don't think we need qbsolv. Kerberos deals with it in more general way

    # def qbsolv(self, fraction=0.1, num_repeats=50, quantum_solver=False):
    #     """ Solves using qbsolv https://readthedocs.com/projects/d-wave-systems-qbsolv/downloads/pdf/latest/
    #     :param fraction:
    #     :param num_repeats:
    #     :param quantum_solver:
    #     :return: tuple (energy, solution)
    #     """
    #     subgraph_size = int(self.size * fraction)
    #     # find embedding of subproblem-sized complete graph to the QPU
    #     if quantum_solver is True:
    #         G = nx.complete_graph(subgraph_size)
    #         system = DWaveSampler()
    #         embedding = minorminer.find_embedding(G.edges, system.edgelist)
    #         solver = FixedEmbeddingComposite(system, embedding)
    #     else:
    #         solver = 'tabu'
    #
    #     # solve a random problem
    #     response = QBSolv().sample_qubo(self.Q,
    #                                     solver=solver,
    #                                     solver_limit=subgraph_size,
    #                                     num_repeats=num_repeats,
    #                                     timeout=30)
    #     best_result = response.data_vectors['energy'], []
    #     for i in range(self.size):
    #         best_result[1].append(int((response.samples()[0][i] + 1) / 2))
    #     self.current_solution = best_result
    #     return self.current_solution

