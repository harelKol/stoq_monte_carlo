import numpy as np 
from PI_path import PI_path_closed
from tqdm import tqdm 
from PI_hamiltonian import Hamiltonian
from multiprocessing import Pool, freeze_support
import os 
import time 



class PI_simulator:
    def __init__(self, H: Hamiltonian, m, b, first_num_mc_iter, num_mc_iter, shift, p_arr, debug=False, save_paths=False):
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
        self.p = p_arr
        self.num_mc_iter = num_mc_iter 
        self.first_num_mc_iter = first_num_mc_iter
        self.path = PI_path_closed(m, H, b, shift, debug=debug)
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
            scheme = np.random.choice([1, 2, 3], p = self.p)
            if scheme == 1:
                self.path.local_update()
            elif scheme == 2:
                self.path.global_update()
            elif scheme == 3:
                self.path.global_update(partial=True)
 
    def calc_op(self, num_samples):
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        avg_op = 0 
        iter = tqdm(range(num_samples))
        for i in iter:
            self.metropolis()
            avg_op += self.path.calc_path_op() / num_samples
            if self.save_paths:
                self.debug_arr.append(self.path.path)
        return avg_op
    
    def calc_op_mp(self, num_samples, num_workers = 8):
        freeze_support()
        num_samples_per_worker = int(num_samples / num_workers)
        inp = [num_samples_per_worker] * num_workers
        with Pool(num_workers) as p:
            try:
                out = p.map(self.calc_op, inp) 
            except KeyboardInterrupt:
                p.terminate()
                p.join()
                exit()
        return np.mean(out, axis = 0)
        
    
    def calc_op_arr(self, num_samples):
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        samples = []
        for i in tqdm(range(num_samples)):
            self.metropolis()
            samples.append(self.path.calc_path_op()) 
        return samples

    def calc_op_arr_mp(self, num_samples, num_workers = 8):
        freeze_support()
        num_samples_per_worker = int(num_samples / num_workers)
        inp = [num_samples_per_worker] * num_workers
        with Pool(num_workers) as p:
            try:
                out = p.map(self.calc_op_arr, inp) 
            except KeyboardInterrupt:
                p.terminate()
                p.join()
                exit()
        arr = []
        for x in out:
            arr += x 
        return np.array(arr)

    
        



    

        
            
            




            
            
