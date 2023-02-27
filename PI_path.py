from PI_hamiltonian import Hamiltonian 
import numpy as np 
from scipy.linalg import expm 
from constants import consts 
import time 

class PI_path_closed:

    def __init__(self, m, H: Hamiltonian, b, debug=False):
        self.debug = debug 
        self.m = m 
        self.H = H
        self.N = self.H.N
        self.num_qubits = H.num_qubits
        self.b = b

        p2 = -(consts.h**2) * (np.diag(np.ones(self.N-1),1) + np.diag(np.ones(self.N-1),-1) -2*np.diag(np.ones(self.N),0)) / (self.H.dx**2) 
        self.exp_c_mat = [expm(-(b/m) * self.H.C_inv[i,i] * p2 / 2) for i in range(self.num_qubits)]

        self.path = np.ones((m + 1, H.num_qubits), dtype = int) * (self.N // 2)
        self.shift = 20
        self.v_weight_arr, self.p_weight_arr = self.calc_weight_arr(self.path)
    
    def local_update(self):
        #local in time
        candidate = np.copy(self.path)
        t = np.random.choice(np.arange(self.m))

        # q = np.random.choice(np.arange(self.num_qubits))
        # candidate[t,q] = (candidate[t,q] + np.random.choice([-self.shift,self.shift]) + self.N) % self.N

        shift_vec = np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits) #make sure no zero vec
        candidate[t,:] = (candidate[t,:] + shift_vec + self.N) % self.N

        if t == 0:
            #keep boundary condition
            candidate[-1,:] = candidate[0,:] 

        W1_curr = self.p_weight_arr[t]
        W2_curr = self.p_weight_arr[t-1]
        W3_curr = self.v_weight_arr[t]
        
        W1_cand = self.calc_p_term(candidate, t)
        W2_cand = self.calc_p_term(candidate, t - 1)
        W3_cand = self.calc_v_term(candidate,t)

        acc_ratio = W1_cand * W2_cand * W3_cand / (W1_curr * W2_curr * W3_curr)

        u = np.random.uniform()
        if self.debug:
            print('local')
            print('t =',t)
            print('curr path =', self.path[0,:])
            print('candidate =', candidate[0,:])
            print('shift =', shift_vec)
            print('acc ratio =', acc_ratio)
            print('u =', u)
        if u < acc_ratio:
            if self.debug:
                print('accepted')
            self.path = candidate 
            self.p_weight_arr[t] = W1_cand 
            self.p_weight_arr[t-1] = W2_cand 
            self.v_weight_arr[t] = W3_cand

    def global_update(self):
        shift_vec = np.tile(np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits), (self.m+1,1))
        candidate = (self.path + shift_vec + self.N) % self.N

        v_w_cand_arr, p_w_cand_arr = self.calc_weight_arr(candidate)
        acc_ratio = np.prod(v_w_cand_arr * p_w_cand_arr / (self.p_weight_arr * self.v_weight_arr))
        u = np.random.uniform()
        if self.debug:
            print('global')
            print('curr path =', self.path[0,:])
            print('candidate =', candidate[0,:])
            print('shift =', shift_vec[0,:])
            print('acc ratio =', acc_ratio)
            print('u =', u)
        if u < acc_ratio:
            if self.debug:
                print('accepted!')
            self.path = candidate 
            self.v_weight_arr, self.p_weight_arr = v_w_cand_arr, p_w_cand_arr

    
    def calc_p_term(self, path ,t):
        state1 = path[t,:]
        if t == -1:
            state2 = path[-2,:]
        else:
            state2 = path[t + 1,:]
        return np.prod([self.exp_c_mat[i][state1[i], state2[i]] for i in range(self.num_qubits)])
    
    def calc_v_term(self, path, t):
        return np.exp(-(self.b / self.m) * self.H.poten(path[t,:]))

    def calc_weight_arr(self, path):
        #the p weight arr is size m array with x(i+1) @ exp(-p2) @ xi 
        #the v weight arr is size m array with exp(-v(xi))
        v_weight_arr = np.zeros(self.m)
        p_weight_arr = np.zeros(self.m)
        for i in range(self.m):
            v_weight_arr[i] = self.calc_v_term(path, i)
            p_weight_arr[i] = self.calc_p_term(path, i)
        return v_weight_arr, p_weight_arr
    
    def calc_path_op(self):
        out = np.mean([self.H.pers_curr(self.path[i]) for i in range(self.m)], axis = 0)
        return out



        
        

        

