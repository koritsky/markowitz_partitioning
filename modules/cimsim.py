# Copyright Russian Quantum Center, 2019
# Dmitry Chermoshentsev, Aleksey Boev

import numpy as np
import torch
import time

class CIMSim:
    
    #Constructor
    def __init__(self, J = np.zeros((0,0)), b = np.zeros(0), make_symmetric = False, quadratic_coeff = 2.0, device='cuda', seed = None):
        if seed != None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        if make_symmetric == True:
            J = 0.5 * (J + J.T)

        self.J = torch.tensor(-J * quadratic_coeff, dtype=torch.float64).to(device)
        self.b = torch.tensor(-b, dtype=torch.float64).to(device)

        self.device = device
        self.dim = self.J.size(0)
        
        if self.dim > 0:
            self.calc_delta(-J * quadratic_coeff, -b)
            
        self.default_params = {'c_th':1., 'zeta':1., 'init_coupling':0.3, 'final_coupling':1., 'N':1000, 'attempt_num':30000, 'dt':0.0001, 'sigma':100., 'alpha':0.9, 'S':1.5, 'D':-0.3, 'O':0.1}
        self.set_params(self.default_params)

    # init_value -- init coupling
    # final_value -- final coupling
    # N -- number of time iterations
    # attempt_num -- number of runs 
    # J -- coupling matrix
    # b -- biases
    # O, S, D -- pump parameters
    # sigma -- sqrt(std) for random number
    # alpha -- momentum parameter
    # c_th -- restriction on amplitudes growth
    def set_params(self, params):
        self.default_params = params
        if 'c_th' in params:
            self.c_th = params['c_th']
        if 'zeta' in params:
            self.zeta = params['zeta']
        if 'init_coupling' in params:
            self.init_coupling = params['init_coupling']
        if 'final_coupling' in params:
            self.final_coupling = params['final_coupling']
        if 'N' in params:
            self.N = params['N']
        if 'attempt_num' in params:
            self.attempt_num = params['attempt_num']
        if 'dt' in params:
            self.dt = params['dt']
        if 'sigma' in params:
            self.sigma = params['sigma']
        if 'alpha' in params:
            self.alpha = params['alpha']
        if 'S' in params:
            self.S = params['S']
        if 'D' in params:
            self.D = params['D']
        if 'O' in params:
            self.O = params['O']
        if 'offset' in params:
            self.offset = params['offset']
        else:
            self.offset = 0
            
    def calc_delta(self, J, b):
        eigenvalues = np.linalg.eigh(J)
        eigsortreverse = torch.tensor(sorted((eigenvalues[0].reshape(-1,1).T + b.T.dot(eigenvalues[1]))[0,:], reverse=True), dtype=torch.float64)

        self.delta = torch.zeros(np.shape(J)[0]-1)
        for i in range(np.shape(J)[0]-1):
            self.delta[i] = eigsortreverse[i] - eigsortreverse[i+1]
        
    # amplitude increment
    def ampl_inc(self, c, zeta, p):
        return (p*c + zeta*(torch.mm(self.J,c)+self.b))*self.dt + (self.sigma*(torch.FloatTensor(c.size(0), self.attempt_num).normal_().to(self.device)).type(torch.float64))*self.dt
    
    def coupling(self):
        self.i = torch.arange(self.N).type(torch.float64).to(self.device)
        self.j = torch.arange(self.dim-1).type(torch.float64).to(self.device)
        time=self.j*self.N*self.dt/(self.dim-1)
        deltsum=torch.sum(self.delta)
        oks=0
        coup=torch.zeros(self.N, dtype=torch.float64).to(self.device)
        for u in range(self.N):
            if oks<self.dim-2:
                if abs(self.dt*u-time[oks+1])<self.dt:
                    oks=oks+1
            coup[u]=coup[u-1]+self.delta[oks]*self.dt*u
        return coup
    
    # pump ramp
    def pump(self, Jmax):
        self.i = torch.arange(self.N, dtype = torch.float64).to(self.device)
        self.arg = torch.tensor(self.S, dtype = torch.float64).to(self.device)*(self.i/self.N-0.5)
        return Jmax*self.O*(torch.tanh(self.arg) + self.D )

    def load_matrix(self, filename, make_symmetric = True, quadratic_coeff = 2.0):
        file = open(filename, 'rb')

        slope_factor = float(file.readline())
        overall_factor = float(file.readline())
        displacement = float(file.readline())
        dt_const = float(file.readline())
        G_in_shape = int(file.readline())
        self.dim = G_in_shape

        h = np.zeros((G_in_shape, 1))
        G = np.zeros((G_in_shape, G_in_shape))

        for i in range(G_in_shape):
            h[i] = file.readline()
            h[i] = -h[i]
            
        G_in = np.loadtxt(file)
        for line in G_in:
            G[int(line[0]-1), int(line[1])-1] = -line[2]
            if (make_symmetric == True):
                G[int(line[1]-1), int(line[0])-1] = -line[2]
        if (make_symmetric == True):
            G = 0.5 * G
            
        G = G * quadratic_coeff

        torch.cuda.empty_cache()
        self.J = torch.tensor(G, dtype=torch.float64).to(self.device)
        self.b = torch.tensor(h, dtype=torch.float64).to(self.device)
        
        self.calc_delta(G, h)

    # evolution of amplitudes
    def evolve(self, callback = None):     
        Jmax = torch.max(torch.sum(torch.abs(self.J),1)).to(self.device)
        
        #random
        random_attempt = np.random.randint(self.attempt_num)

        # initializing current amplitudes
        c = torch.zeros((self.dim, self.attempt_num), dtype=torch.float64, device=self.device)
        spins = torch.sign(c)
        dc = torch.zeros_like(c).to(self.device)
        
        # creating the array for evolving amplitudes from random attempt
        c_evol = torch.empty((self.dim, self.N), dtype=torch.float64, device=self.device)
        c_evol[:,0] = c[:,random_attempt]
        # define pump array
        p = self.pump(Jmax).to(self.device)
        H_opt = -0.5 * torch.einsum('ij,ik,jk->k', (self.J, spins, spins)) - torch.einsum('ij,ik->k', (self.b, spins)) + self.offset
        
        c_opt = c
        
        zeta = self.coupling()
        
        # initializing moving average of amplitudes increment
        dc_momentum = torch.zeros((self.dim, self.attempt_num), dtype=torch.float64, device=self.device)
        #c_all=torch.zeros((dim, attempt_num, N), dtype=torch.float64)
        for i in range(1, self.N):

            # calculating amplitude increment
            dc = self.ampl_inc(c, zeta[i], p[i])

            # calculating moving average of amplitudes increment
            dc_momentum = self.alpha * dc_momentum + (1 - self.alpha) * dc

            # calculating possible values of amplitudes on the next step
            c1 = c + dc_momentum

            # updating c
            c = torch.tanh(c + dc_momentum)
            spins = torch.sign(c)
            H = -0.5 * torch.einsum('ij,ik,jk->k',(self.J, spins, spins)) - torch.einsum('ij,ik->k',(self.b, spins)) + self.offset
            check = (H < H_opt).type(torch.float64)
            H_opt = check * H + (1 - check) * H_opt
            c_opt = check * c + (1 - check) * c_opt
            
            # add amplitude values from random attempt to c_evol array 
            c_evol[:,i] = c[:, random_attempt]
            
            if (( (hasattr(self, 'time_current') == False) or (time.time() - self.time_current > 10) or (i == self.N-1) ) and (callback != None)):
                self.time_current = time.time()
                spins_ising = np.zeros(self.N)
                spins_current = torch.sign(c_opt)
                if self.device == "cuda":
                    spins_ising = spins_current[:,torch.argmin(H_opt)].cpu().data.numpy().tolist()
                else:
                    spins_ising = spins_current[:,torch.argmin(H_opt)].numpy().tolist()
                callback(spins_ising)
            
        self.lastcuts = H_opt
        return c, c_evol, self.lastcuts, c_opt
    
    def find_opt(self, callback = None):
        self.time_current = time.time() 
        c, c_evol, H_opt, c_opt = self.evolve(callback)
        spins_current = torch.sign(c_opt)
        spins_ising = np.zeros(self.N)
        
        if self.device == "cuda":
            spins_ising = spins_current[:,torch.argmin(H_opt)].cpu().data.numpy().tolist()
        else:
            spins_ising = spins_current[:,torch.argmin(H_opt)].numpy().tolist()
            
        energy_ising = H_opt.min()
        return (spins_ising, energy_ising, c, c_evol)


class ising_utilits:
    @staticmethod
    def to_ising_file(h, J, filename):
        """ Create an ising .txt file, that contains linear and quadratic coeffients as well as other (?) information

        :param h: list of linear coefficients
        :param J: matrix of quadratic coefficients
        :param filename: name of file
        """
        n = len(h)
        f = open(filename, 'w')
        f.write("%f\n" % 3.0)
        f.write("%f\n" % 0.1)
        f.write("%f\n" % -0.9)
        f.write("%f\n" % 0.07)
        f.write("%d\n" % n)
        for i in range(n):
            if i in h.keys():
                f.write("%f\n" % h[i])
            else:
                f.write("%f\n" % 0)
        for row in range(n):
            for col in range(row + 1, n):
                if (row, col) in J.keys():
                    f.write("%d %d %f\n" % (row + 1, col + 1, J[row, col]))
                else:
                    f.write("%d %d %f\n" % (row + 1, col + 1, 0))

    @staticmethod
    def ising_to_matrix(h, J):
        """ Transfrom dictionaries into vector and matrix for ising task.

        :param h: dict of linear coefficients
        :param J: dict of quadratic coefficients
        :return: tuple (hvector, Jmatrix) - vector of linear and matrix of quadratic coefficients
        """
        n = len(h)

        hvector = np.zeros((n, 1))
        for i in range(n):
            if i in h.keys(): hvector[i] = h[i]

        Jmatrix = np.zeros((n, n))
        for row in range(n):
            for col in range(row + 1, n):
                if (row, col) in J.keys(): Jmatrix[row, col] = J[(row, col)]
        return hvector, Jmatrix
