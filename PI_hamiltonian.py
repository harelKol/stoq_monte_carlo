from constants import consts
import numpy as np 

class Hamiltonian:
    def __init__(self, C_inv, L_inv, Ej, x0, x_max, N):
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
        self.v_const = self.find_v_const()
    
    def find_v_const(self):
        # return 0 * self.num_qubits * (np.max(np.diag(self.L_inv)) * ((self.x_max - np.min(self.x0)) ** 2) / 2 + np.max(self.Ej))
        return 1500
   
    def poten(self,state):
        x = self.axis[state]
        x_tag = x - self.x0
        harm = x_tag @ self.L_inv @ x_tag / 2
        un_harm = 0
        for i in range(self.num_qubits):
            un_harm += self.Ej[i] * np.cos(2*np.pi*x[i]/consts.p0)
        return harm + un_harm - self.v_const
    
    def pers_curr(self, state):
        x = self.axis[state]
        x_tag = x - self.x0
        return self.L_inv[0,:] @ x_tag * (1e6 / consts.mult_const) 
        #qubit 0 

    # def op(self,state):
    #     x = self.axis[state]
    #     x_tag = x - self.x0
    #     harm = x_tag @ self.L_inv @ x_tag 
    #     un_harm1 = 0
    #     un_harm2 = 0 
    #     for i in range(self.num_qubits):
    #         un_harm1 -= self.Ej[i] * x[i] * np.pi * np.sin(2*np.pi*x[i]/consts.p0) / consts.p0
    #         un_harm2 += self.Ej[i] * np.cos(2*np.pi*x[i]/consts.p0)
    #     return harm + un_harm1 + un_harm2
    
    # def kin(self, state1, state2):
    #     a = (state1 == state2).astype(float)
    #     b = (state1 == (state2 + 1) % self.N).astype(float) #add modulo!!!!!!
    #     c = (state1 == (state2 - 1 + self.N) % self.N).astype(float)
    #     out = 0 
    #     for i in range(self.num_qubits):
    #         #i=j
    #         temp = np.copy(a)
    #         temp[i] = 1 
    #         out += self.C_inv[i,i] * (2*a[i] - b[i] - c[i]) * np.prod(temp) / 2 
            
    #         for j in range(i+1,self.num_qubits):
    #             temp = np.copy(a)
    #             temp[i] = 1 
    #             temp[j] = 1 
    #             out -= self.C_inv[i,j] * (b[i] * b[j] + c[i] * c[j] - c[i] * b[j] - c[j] * b[i]) * np.prod(temp) / 4 
    #     out *= (consts.h / self.dx)**2

    #     return out 

    # def h(self, state1, state2):
    #     a = (state1 == state2).astype(int)
    #     k = self.kin(state1, state2)
    #     v = self.poten(state1) 
    #     h = k + v * np.prod(a)
    #     return h

    


            

        








    



        