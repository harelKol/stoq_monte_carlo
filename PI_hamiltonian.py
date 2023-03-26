from constants import consts
import numpy as np 
from scipy.linalg import sqrtm

class Hamiltonian:
    def __init__(self, C_inv, L_inv, Ej, x0, x_max, N, transform = False):
        '''
        C_inv, L_inv, Ej - circuit params 
        x_max, Ns - discretization params 
        p_constant - positive constant insure positivity 
        '''
        self.C_inv = C_inv 
        self.L_inv = L_inv 
        self.Ej = Ej 
        self.x0 = x0 
        self.x_max = x_max 
        self.N = N
        self.axis = np.linspace(-x_max, x_max, N)
        self.dx = 2 * x_max / (N - 1)
        self.num_qubits = C_inv.shape[0]
        self.transform = transform 
        if transform:
            # sqrt_c = sqrtm(C_inv)
            # const = np.sqrt(np.trace(C_inv @ C_inv))
            # self.S = sqrt_c / np.sqrt(const)
            # self.C_inv = np.eye(self.num_qubits) * const 
            # self.L_inv = self.S @ self.L_inv @ self.S
            # self.S_inv = np.linalg.inv(self.S)

            E, V = np.linalg.eigh(C_inv)
            self.C_inv = np.diag(E)
            self.V = V 
            self.L_inv = V.T @ self.L_inv @ V 
            self.x0 = V.T @ self.x0
        
   
    def poten(self,state):
        x = self.axis[state]
        x_tag = x - self.x0
        harm = x_tag @ self.L_inv @ x_tag / 2
        un_harm = 0
        if self.transform:
            # x = self.S @ x 
            x = self.V @ x 
        for i in range(self.num_qubits):
            un_harm += self.Ej[i] * np.cos(2*np.pi*x[i]/consts.p0)
        return harm + un_harm 
    
    def pers_curr(self, state):
        x = self.axis[state]
        x_tag = x - self.x0
        if self.transform:
            # return self.S_inv @ self.L_inv @ x_tag * (1e6 / consts.mult_const) 
            return self.V @ self.L_inv @ x_tag * (1e6 / consts.mult_const) 
        return self.L_inv @ x_tag * (1e6 / consts.mult_const) 
    
    def poten_min(self, x):
        x_tag = x - self.x0
        harm = x_tag @ self.L_inv @ x_tag / 2
        un_harm = 0
        if self.transform:
            # x = self.S @ x 
            x = self.V @ x 
        for i in range(self.num_qubits):
            un_harm += self.Ej[i] * np.cos(2*np.pi*x[i]/consts.p0)
        return harm + un_harm 

    
    def pers_curr_min(self, x):
        x_tag = x - self.x0
        if self.transform:
            # return self.S_inv @ self.L_inv @ x_tag * (1e6 / consts.mult_const) 
            return self.V @ self.L_inv @ x_tag * (1e6 / consts.mult_const) 
        return self.L_inv @ x_tag * (1e6 / consts.mult_const) 


    


            

        








    



        