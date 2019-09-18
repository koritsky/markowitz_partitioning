import dimod
import numpy as np
import random
from hybrid.reference.kerberos import KerberosSampler
import pyeasyga
from modules.cplex_solver import CplexSolver
from modules.cimsim import CIMSim
from modules.ising_utilits import ising_utilits


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
        if qubo_mat is not None:
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
        solution = np.array([int((i + 1) / 2) for i in response.first[0].values()])
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

    def cim_solver(self, params: dict):
        h, J, _ = self.bqm.to_ising()
        hvector, Jmatrix = ising_utilits.ising_to_matrix(h, J)

        # Generate "ising file" for ParameterOptimizer
        ising_utilits.to_ising_file(hvector, Jmatrix, "block_4_ising.txt")

        # Run CIMSim
        cimsim = CIMSim(Jmatrix, hvector, device="cpu")
        cimsim.set_params(params = params)
        (spins_ising, energy_ising, c_current, c_evol) = cimsim.find_opt()

        # Print solution as permutation matrix
        solution = np.array([int((i + 1) / 2) for i in spins_ising])
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution

    def kerberos_solver(self, **kwargs):
        """Run Tabu search, Simulated annealing and QPU subproblem sampling (for
            high energy impact problem variables) in parallel and return the best
            samples.

            Sampling Args:

                bqm (:obj:`~dimod.BinaryQuadraticModel`):
                    Binary quadratic model to be sampled from.

                init_sample (:class:`~dimod.SampleSet`, callable, ``None``):
                    Initial sample set (or sample generator) used for each "read".
                    Use a random sample for each read by default.

                num_reads (int):
                    Number of reads. Each sample is the result of a single run of the
                    hybrid algorithm.

            Termination Criteria Args:

                max_iter (int):
                    Number of iterations in the hybrid algorithm.

                max_time (float/None, optional, default=None):
                    Wall clock runtime termination criterion. Unlimited by default.

                convergence (int):
                    Number of iterations with no improvement that terminates sampling.

                energy_threshold (float, optional):
                    Terminate when this energy threshold is surpassed. Check is
                    performed at the end of each iteration.

            Simulated Annealing Parameters:

                sa_reads (int):
                    Number of reads in the simulated annealing branch.

                sa_sweeps (int):
                    Number of sweeps in the simulated annealing branch.

            Tabu Search Parameters:

                tabu_timeout (int):
                    Timeout for non-interruptable operation of tabu search (time in
                    milliseconds).

            QPU Sampling Parameters:

                qpu_reads (int):
                    Number of reads in the QPU branch.

                qpu_sampler (:class:`dimod.Sampler`, optional, default=DWaveSampler()):
                    Quantum sampler such as a D-Wave system.

                qpu_params (dict):
                    Dictionary of keyword arguments with values that will be used
                    on every call of the QPU sampler.

                max_subproblem_size (int):
                    Maximum size of the subproblem selected in the QPU branch.

                Returns:
                    :obj:`~dimod.SampleSet`: A `dimod` :obj:`.~dimod.SampleSet` object.

                """
        response = KerberosSampler().sample(self.bqm, **kwargs)
        solution = np.array([int(i) for i in response.first[0].values()])
        self.current_solution = self.solution_energy(solution), solution
        return self.current_solution

    def ga_solver(self,
                  iteration_number=30,
                  population_size=30,
                  generations=100,
                  init_solution=None,
                  ):
        """Solves using genetic algorithm
        :param iteration_number: Number of iteration of genetic algorithm
        :param population_size:
        :param generations:
        :param init_solution: list List of initial "good" solutions to start with
        :return: tuple (energy, solution)
        """

        data = self.bqm
        ga = pyeasyga.GeneticAlgorithm(data,
                                       population_size=population_size,
                                       generations=generations,
                                       maximise_fitness=False,
                                       elitism=True)

        def create_individual(data):
            return [random.randint(0, 1) for _ in range(self.size)]

        def fitness(individual, data):
            return dimod.BinaryQuadraticModel.energy(data, individual)

        ga.fitness_function = fitness
        ga.create_individual = create_individual
        ga.create_initial_population(initial_genes=init_solution)
        best_result = None, []
        for i in range(iteration_number):
            ga.run()
            if best_result[0] is None or ga.best_individual()[0] < best_result[0]:
                best_result = ga.best_individual()
        solution = best_result[1]
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
