from __future__ import print_function
import numpy as np
import dimod
from binary_problem import BinaryProblem


class Portfolio(BinaryProblem):
    def __init__(self,
                 budget=0,
                 theta=None,
                 prices=None,
                 averages=None,
                 covariance=None,):
        super().__init__(bqm=None, qubo_mat=None)
        if covariance is None:
            covariance = np.array([])
        if averages is None:
            averages = np.array([])
        if prices is None:
            prices = np.array([])
        if theta is None:
            theta = [0.3, 0.3, 0.3]
        self.prices = prices
        self.averages = averages
        self.covariance = covariance
        self.budget = budget
        self.theta = theta

    def file_read(self, file_prices, file_averages, file_covariance, threshold=0):
        """ Reads prices, averages and covariance from csv files
        :param file_prices:
        :param file_averages:
        :param file_covariance:
        :param threshold: If != 0, safe only up to threshold number of variables
        """
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
        self.to_qubo()

    def to_markowitz_qubo(self):
        """ Generates QUBO model by markowitz problem and, based on it, dimod.bqm model.
        """
        assert not np.array_equal(self.averages, []), "No averages added"
        assert not np.array_equal(self.prices, []), "No prices added"
        assert not np.array_equal(self.covariance, []), "No covariance added"

        for row in range(self.size):
            self.qubo_mat[row, row] = (self.theta[1] * self.covariance[row][row]
                                       + self.theta[2] * self.prices[row] ** 2
                                       - self.theta[0] * self.averages[row]
                                       - 2 * self.budget * self.theta[2] * self.prices[row])
            for col in range(row + 1, self.size):
                self.qubo_mat[row, col] = 2 * (self.theta[1] * self.covariance[row][col]
                                               + self.theta[2] * self.prices[row] * self.prices[col])
        self.from_qubo(self.qubo_mat)  # Builds a BQM model
        return self.qubo_mat

    def results(self, solution=None):
        """ Count and returns portfolio return and variance based on given solution
        :param solution:
        """
        if solution is None:
            solution = self.current_solution[1]
        solution = np.asarray(solution)
        portfolio_return = np.dot(solution, self.averages)
        portfolio_variance = np.dot(solution, np.dot(self.covariance, solution))
        return portfolio_return, portfolio_variance
