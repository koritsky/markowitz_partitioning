from hybrid.core import Runnable
from modules.binary_problem import BinaryProblem
import numpy as np
import hybrid
import dimod
from modules.cimsim import CIMSim, ising_utilits


def simcim_solver(bqm):
    h, J = bqm.linear, bqm.quadratic
    hvec, Jmat = ising_utilits.ising_to_matrix(h, J)

    # Following 2 lines are responsible for getting solution from external simcim solver
    cimsim = CIMSim(Jmat, hvec.reshape(-1, 1), device='cpu')
    cimsim.attempt_num = 100

    spins_ising, energy_ising, c_current, c_evol = cimsim.find_opt()
    return spins_ising

def sample_to_array(sample):
    try:
        return [int((i + 1) / 2) for i in sample.first[0].values()]
    except TypeError:
        print("sample type is not SampleSet")
        return sample


class SimCIMSampler(Runnable):

    def __init__(self, **runopts):
        super().__init__(**runopts)

    def next(self, state, **runopts):
        bqm = state.subproblem

        # Just a ndarray of zeros and ones:
        raw_solution = simcim_solver(bqm)
        solution = {i: j for i, j in zip(bqm.linear.keys(), raw_solution)}

        response = dimod.SampleSet.from_samples_bqm(solution, bqm=bqm)

        return state.updated(subsamples=response)

