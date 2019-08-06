# Copyright Russian Quantum Center, 2019
# Aleksey Boev

import cplex
from cplex.exceptions import CplexSolverError
import docloud.job

import numpy as np
import xml.etree.ElementTree as ET

import os
import utilits


class CplexSolver:
    default_model_file = "model.lp"
    default_solution_file = "solution.xml"
    default_log_file = "log.xml"

    def generate_model(self, print_log=False):

        variable_product_ids = set({})
        variable_ids = set({})
        coefficient_lookup = {}

        for i in range(self.N):
            variable_ids.add(i)
            coefficient_lookup[(i, i)] = self.qubo[i, i]

        for i in range(self.N):
            for j in range(i + 1, self.N, 1):
                coefficient_lookup[(i, j)] = self.qubo[i, j]
                variable_product_ids.add((i, j))
        self.model = cplex.Cplex()

        # print(coefficient_lookup)
        self.variable_lookup = {}
        for vid in variable_ids:
            coeff = 0
            if (vid, vid) in coefficient_lookup:
                coeff = coefficient_lookup[(vid, vid)]

            indexes = self.model.variables.add(obj=[coeff], lb=[0], ub=[1], types=['B'], names=['s' + str(vid)])

            assert (len(indexes) == 1)
            self.variable_lookup[(vid, vid)] = next((x for x in indexes))

        for pair in variable_product_ids:
            indexes = self.variable_lookup[pair] = self.model.variables.add(
                obj=[coefficient_lookup[pair]],
                lb=[0], ub=[1], types=["B"])
            assert (len(indexes) == 1)
            self.variable_lookup[pair] = next((x for x in indexes))

        for i, j in variable_product_ids:
            terms = cplex.SparsePair(
                ind=[self.variable_lookup[(i, j)], self.variable_lookup[(i, i)], self.variable_lookup[(j, j)]],
                val=[1, -1, -1])
            self.model.linear_constraints.add(lin_expr=[terms], senses=['G'], rhs=[-1])

            terms = cplex.SparsePair(ind=[self.variable_lookup[(i, j)], self.variable_lookup[(i, i)]], val=[1, -1])
            self.model.linear_constraints.add(lin_expr=[terms], senses=['L'], rhs=[0])

            terms = cplex.SparsePair(ind=[self.variable_lookup[(i, j)], self.variable_lookup[(j, j)]], val=[1, -1])
            self.model.linear_constraints.add(lin_expr=[terms], senses=['L'], rhs=[0])

        self.model.objective.set_sense(self.model.objective.sense.minimize)

    def load_solution(self, filename, dims):
        f = open(filename, 'r')
        spins_qubo = np.zeros(dims)
        f.seek(0)
        context = ET.iterparse(f, events=("start", "end"))
        for event, elem in context:
            if (elem.tag == "variable") and (elem.attrib['name'][0] == 's'):
                idx = int(elem.attrib['name'][1:])
                spins_qubo[idx] = elem.attrib['value']
            elif elem.tag == "header":
                print(elem.attrib['solutionStatusString'] + " (" + elem.attrib['solutionStatusValue'] + ")")
                # Details on https://www.hpc.science.unsw.edu.au/files/docs/ilog/cplex/12.1/html/Content/Optimization/Documentation/CPLEX/_pubskel/cplex_matlab1234.html
        return spins_qubo

    def find_opt(self, solve_local=False, print_log=False, waittime=-1):
        spins_qubo = []
        if solve_local:
            try:
                timestart = self.model.get_time()
                self.model.solve()
                runtime = self.model.get_time() - timestart
            except CplexSolverError as e:
                print("Exception raised during solve: " + e)
            else:
                solution = self.model.solution
                if print_log:
                    print(solution.get_quality_metrics())
                    print('status: {} - {}'.format(solution.get_status(), solution.get_status_string()))
                    print('   gap: {}'.format(solution.MIP.get_mip_relative_gap()))

                spins_qubo = np.zeros(self.N)
                for k in sorted(self.variable_lookup):
                    v = self.variable_lookup[k]
                    if solution.get_values(v) != 0.0:
                        if k[0] == k[1]:
                            spins_qubo[k[0]] = 1
                        if print_log:
                            print('{}: {}'.format(k, solution.get_values(v)))

                upper_bound = solution.get_objective_value()
                lower_bound = upper_bound * (1 + solution.MIP.get_mip_relative_gap())

        else:
            if not os.path.isfile(self.cplex_solution_file):
                self.model.write(self.cplex_model_file)
                client = docloud.job.JobClient(url=self.url, api_key=self.api_key)
                if waittime == -1:
                    response = client.execute(input=[self.cplex_model_file], output=self.cplex_solution_file, gzip=True,
                                              log=self.cplex_log_file)
                    spins_qubo = self.load_solution(self.cplex_solution_file, dims=self.N)
                else:
                    try:
                        response = client.execute(input=[self.cplex_model_file], output=self.cplex_solution_file,
                                                  gzip=True, log=self.cplex_log_file, waittime=waittime,
                                                  continuous_logs=True, parameters={'oaas.timeLimit': waittime})
                        spins_qubo = self.load_solution(self.cplex_solution_file, dims=self.N)
                    except:
                        spins_qubo = self.load_solution(self.cplex_solution_file, dims=self.N)
            else:
                spins_qubo = self.load_solution(self.cplex_solution_file, dims=self.N)
            if (self.cplex_model_file == self.default_model_file) and (os.path.isfile(self.cplex_model_file)):
                os.remove(self.cplex_model_file)
            if ((self.cplex_solution_file == self.default_solution_file) and (
                    os.path.isfile(self.cplex_solution_file))):
                os.remove(self.cplex_solution_file)

        return [int(i) for i in spins_qubo]

    def __init__(self,
                 qubo_matrix,
                 model_file=default_model_file,
                 solution_file=default_solution_file,
                 log_file=default_log_file,
                 url="",
                 api_key=""):
        self.qubo = qubo_matrix
        self.N = self.qubo.shape[0]

        self.url = url
        self.api_key = api_key

        self.cplex_model_file = model_file
        self.cplex_solution_file = solution_file
        self.cplex_log_file = log_file

        self.generate_model()


if __name__ == "__main__":
    block_dim = [5, 5, 5]
    np.random.seed(0)
    dim = sum(block_dim)
    mixed_matrix = utilits.mixed_matrix_generator(block_dim)
    qubo_matrix = utilits.partitioning_qubo_build(mixed_matrix)
    cplexsolver = CplexSolver(qubo_matrix=qubo_matrix,
                              url="https://api-oaas.docloud.ibmcloud.com/job_manager/rest/v1/",
                              api_key='api_642ddfbf-ae49-4947-84f1-196f6883eab2')
    qubo_solution = cplexsolver.find_opt()
    solution_permutation_matrix = np.array(qubo_solution).reshape((dim, dim))
    if not utilits.permutation_check(solution_permutation_matrix):
        print('\033[93m' + "Solution is not permutation matrix" + '\033[0m')
    print(solution_permutation_matrix)

    new_matrix = np.dot(solution_permutation_matrix.T,
                        np.dot(mixed_matrix, solution_permutation_matrix))
    print(new_matrix)
