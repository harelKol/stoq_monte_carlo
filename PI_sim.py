import numpy as np 
from PI_path import PI_path_closed
from tqdm import tqdm 
from PI_hamiltonian import Hamiltonian


class PI_simulator:
    def __init__(self, H: Hamiltonian, m, b, num_samples, first_num_mc_iter, num_mc_iter, debug=False, save_paths=False):
        '''
        should be additional simulation params
        H : Hamiltonian class 
        m : number of operators product 
        b : inverse temperature 
        '''
        self.debug = debug 
        self.save_paths = save_paths
        self.H = H 
        self.m = m
        self.b = b
        self.num_samples = num_samples 
        self.num_mc_iter = num_mc_iter 
        self.first_num_mc_iter = first_num_mc_iter
        self.path = PI_path_closed(m, H, b, debug=debug)
        self.samples = []
        self.debug_arr = []
        self.metropolis(first = True)
        print('--- Finished First Iters ---')
    
    def metropolis(self, first=False):
        # self.debug_op = []
        if first:
            iter = tqdm(range(self.first_num_mc_iter))
        else:
            iter = range(self.num_mc_iter)
        for i in iter:
            if self.debug:
                x = input()
            scheme = np.random.choice([1, 2], p = [0.9, 0.1])
            if scheme == 1:
                self.path.local_update()
            elif scheme == 2:
                self.path.global_update()
 
    def calc_op(self, k = None):
        avg_op = 0 
        if k is None:
            iter = tqdm(range(self.num_samples))
        else:
            iter = range(self.num_samples)
        for i in iter:
            self.metropolis()
            avg_op += self.path.calc_path_op() / self.num_samples
            if self.save_paths:
                self.debug_arr.append(self.path.path)
        return avg_op
    
    def calc_op_arr(self):
        self.samples = []
        for i in tqdm(range(self.num_samples)):
            self.metropolis()
            self.samples.append(self.path.calc_path_op()) 
    
        



    

        
            
            




            
            
