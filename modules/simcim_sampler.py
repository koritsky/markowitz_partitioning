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


class SimCIMSampler(Runnable):

    def __init__(self, **runopts):
        super().__init__(**runopts)
        self.tuner_timeout = runopts['tuner_timeout']
        self.simcim_attempt_num = runopts['simcim_attempt_num']

    # function is called on each iteration of branch.
    def next(self, state, **runopts):
        # form a bqm from subproblem that was determined by EnergyDecomposer.

        self.bqm = state.subproblem
        # just a ndarray of zeros and ones:
        raw_solution = self.simcim_solver()

        # here we take care about the indexes, because we need to update correct ones in general problem
        solution = {i: j for i, j in zip(self.bqm.linear.keys(), raw_solution)}
        response = dimod.SampleSet.from_samples_bqm(solution, bqm=self.bqm)

        # return to composer the state which has parameter "subsamples" with our response
        return state.updated(subsamples=response)

    def simcim_solver(self):
        # formalize problem as ising and qubo model respectively
        h, J = ising_utilits.ising_to_matrix(self.bqm.linear, self.bqm.quadratic)

        # upload solutions
        solutions = qboard.cache.Solutions()

        "turn this on if you want not to run simcim if this qubo was already solved"
        # if (h, J) in solutions:
        #     spins_ising = [1 if _ else -1 for _ in solutions[h, J]]
        #     energy_ising = qubo.ienergy(h, J, spins_ising)
        # else:

        # run parameter tuner for no longer than "timeout" seconds
        params = qboard.cache.Parameters()

        def optimizer(h, J):
            params[h, J] = tuner.optimize(h, J)

        p = multiprocessing.Process(target=optimizer, name="Optimizer", args=(h, J))
        p.start()
        p.join(self.tuner_timeout)
        if p.is_alive():
            p.terminate()
            p.join()

        # run "simcim_attempt_num" iterations of cimsim
        cimsim = CIMSim(J, h.reshape(-1, 1), device='cpu')
        cimsim.set_params({'c_th': 1.,
                           'zeta': 1.,
                           'init_coupling': 0.3,
                           'final_coupling': 1.,
                           'N': 1000,
                           'attempt_num': self.simcim_attempt_num,
                           **params[h, J]})
        spins_ising, energy_ising, c_current, c_evol = cimsim.find_opt()
        solutions[h, J] = spins_ising

        return spins_ising