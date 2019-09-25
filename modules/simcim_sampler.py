"""
This class is responsible for sending subproblems to cimsim.
"""
import multiprocessing
import dimod
from hybrid.core import Runnable
from qboard.cimsim.cimsim import CIMSim
from qboard.cimsim import tuner
import qboard
import qboard.cache
from modules.ising_utilits import ising_utilits
from modules.binary_problem import BinaryProblem


class SimCIMSampler(Runnable):

    def __init__(self, **runopts):
        super().__init__(**runopts)
        self.tuner_timeout = runopts['tuner_timeout']
        self.simcim_attempt_num = runopts['simcim_attempt_num']

    # function is called on each iteration of branch.
    def next(self, state, **runopts):
        # form a bqm from subproblem that was determined by EnergyDecomposer.
        self.bqm = state.subproblem
        problem = BinaryProblem(bqm=self.bqm)
        _, raw_solution = problem.simcim_solver(tuner_timeout=self.tuner_timeout,
                                                attempt_num=self.simcim_attempt_num)
        # here we take care about the indices, because we need to update only correct ones in general problem
        solution = {i: j for i, j in zip(self.bqm.linear.keys(), raw_solution)}
        response = dimod.SampleSet.from_samples_bqm(solution, bqm=self.bqm)

        # return to composer the state which has parameter "subsamples" with our response
        return state.updated(subsamples=response)
