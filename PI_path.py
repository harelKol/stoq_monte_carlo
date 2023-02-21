from PI_hamiltonian import Hamiltonian 
import numpy as np 
from scipy.linalg import expm 
from constants import consts 
import time 
def truncated_prod(arr):
    out = 1 
    for x in arr:
        out *= x 
        if out > 1:
            return 1 
        if out < 1e-16:
            return out 
    return out 
class path_class:
    def __init__(self, m, H: Hamiltonian, b):
        #a path is m + 1 X num_qubits array
        self.m = m 
        self.H = H
        self.N = self.H.N
        self.b = b
        self.path = np.tile(np.random.choice(np.arange(self.N), size=H.num_qubits), (m,1))
        self.shift = 10
        self.curr_weight_arr = self.calc_weight_arr(self.path)

    def local_update(self):
        #local in time
        candidate = np.copy(self.path)
        t = np.random.choice(np.arange(self.m))
        shift_vec = np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits)
        candidate[t,:] = (candidate[t,:] + shift_vec + self.N) % self.N
        # if t == 0:
        #     #keep boundary condition
        #     candidate[-1,:] += shift_vec 
        if t == 0: 
            W1_cand = self.calc_partial_weight(candidate, t)
            W1_curr = self.curr_weight_arr[t]
            acc_ratio = W1_cand / W1_curr 
        elif t == (self.m - 1):
            W2_cand = self.calc_partial_weight(candidate, t - 1)
            W2_curr = self.curr_weight_arr[t-1]
            acc_ratio = W2_cand / W2_curr 
        else:
            W1_cand = self.calc_partial_weight(candidate, t)
            W2_cand = self.calc_partial_weight(candidate, t - 1)
            W1_curr = self.curr_weight_arr[t]
            W2_curr = self.curr_weight_arr[t-1]
            acc_ratio = W1_cand * W2_cand / (W1_curr * W2_curr)

        u = np.random.uniform()
        if u < acc_ratio:
            self.path = candidate 
            if t == 0:
                self.curr_weight_arr[t] = W1_cand 
            elif t == self.m - 1:
                self.curr_weight_arr[t-1] = W2_cand 
            else:
                self.curr_weight_arr[t] = W1_cand 
                self.curr_weight_arr[t-1] = W2_cand 

    def global_update(self):
        shift_vec = np.tile(np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits), (self.m,1))
        candidate = (self.path + shift_vec + self.N) % self.N
        W_candidate_arr = self.calc_weight_arr(candidate)
        acc_ratio = np.prod(W_candidate_arr) / np.prod(self.curr_weight_arr)
        u = np.random.uniform()
        if u < acc_ratio:
            self.path = candidate 
            self.curr_weight_arr = W_candidate_arr 
     
    def calc_partial_weight(self, path, t):
        state1 = path[t,:]
        state2 = path[t + 1,:]
        I = np.prod((state1 == state2).astype(float))
        W_curr = I - (self.b/self.m) * self.H.h(state1, state2)
        return W_curr
    
    def calc_weight_arr(self, path):
        W_arr = np.zeros(self.m - 1)
        for t in range(self.m - 1):
            W_curr = self.calc_partial_weight(path, t)
            W_arr[t] = W_curr 
        return W_arr 
    
    def calc_path_E(self):
        state1 = self.path[0,:]
        state2 = self.path[-1,:]
        return self.H.h(state1, state2)
    
    # def calc_op(self):
    #     return self.H.op(self.path[0,:]) 


class path_closed:
    def __init__(self, m, H: Hamiltonian, b, debug=False):
        #a path is m + 1 X num_qubits array
        self.m = m 
        self.H = H
        self.N = self.H.N
        self.num_qubits = H.num_qubits
        self.b = b
        # self.path = np.tile(np.random.choice(np.arange(self.N), size=self.num_qubits), (m+1,1))
        self.path = np.ones((m + 1, H.num_qubits), dtype = int) * (self.N // 2)
        self.shift = 1
        self.curr_weight_arr = self.calc_weight_arr(self.path)
        self.debug = debug 

    def local_update(self):
        #local in time
        candidate = np.copy(self.path)
        t = np.random.choice(np.arange(self.m))
        # shift_vec = np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits)
        q = np.random.choice(np.arange(self.H.num_qubits))
        # candidate[t,:] = (candidate[t,:] + shift_vec + self.N) % self.N
        candidate[t,q] = (candidate[t,q] + np.random.choice([-1,1]) + self.N) % self.N
        if t == 0:
            #keep boundary condition
            candidate[-1,:] = candidate[0,:] 
        W1_cand = self.calc_partial_weight(candidate, t)
        W2_cand = self.calc_partial_weight(candidate, t - 1)
        W1_curr = self.curr_weight_arr[t]
        W2_curr = self.curr_weight_arr[t-1]
        acc_ratio = W1_cand * W2_cand / (W1_curr * W2_curr)
        u = np.random.uniform()
        if self.debug:
            time.sleep(3)
            print('t =',t)
            print('curr path =', self.path)
            print('candidate =', candidate)
            print('acc ratio =', acc_ratio)
            print('u =', u)

        if u < acc_ratio:
            if self.debug:
                print('accepted')
            self.path = candidate 
            self.curr_weight_arr[t] = W1_cand 
            self.curr_weight_arr[t-1] = W2_cand 


    def global_update(self):
        shift_vec = np.tile(np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits), (self.m+1,1))
        candidate = (self.path + shift_vec + self.N) % self.N
        W_candidate_arr = self.calc_weight_arr(candidate)
        acc_ratio = truncated_prod(W_candidate_arr / self.curr_weight_arr)
        u = np.random.uniform()
        if self.debug:
            time.sleep(5)
            print('curr path =', self.path[0,:])
            print('candidate =', candidate[0,:])
            print('shift =', shift_vec[0,:])
            print('acc ratio =', acc_ratio)
            print('u =', u)
        if u < acc_ratio:
            if self.debug:
                print('accepted!')
            self.path = candidate 
            self.curr_weight_arr = W_candidate_arr 
     
    def calc_partial_weight(self, path, t):
        state1 = path[t,:]
        if t == -1:
            state2 = path[-2,:]
        else:
            state2 = path[t + 1,:]
        I = np.prod((state1 == state2).astype(float))
        W_curr = I - (self.b/self.m) * self.H.h(state1, state2)
        # W_curr = np.exp(- (self.b/self.m) * self.H.h(state1, state2))
        if W_curr < 0:
            print(W_curr)
            raise ValueError('negative Weight')
        return W_curr
    
    def calc_weight_arr(self, path):
        # with multiprocessing.Pool(processes=self.m) as pool:
        #     W_arr = pool.starmap(self.calc_partial_weight, zip(repeat(path),range(self.m)))
        # W_arr = np.array(W_arr)
        W_arr = np.zeros(self.m)
        for t in range(self.m):
            W_arr[t] = self.calc_partial_weight(path, t)
        return W_arr 
    
    def calc_path_E(self):
        # if self.path[0,:] != self.path[-1,:]:
        #     raise ValueError('no boundary conditions')
        state = self.path[0,:]
        out = self.H.pers_curr(state)
        # out = 0 
        # for i in range(self.m):
        #     out += (self.H.pers_curr(self.path[i,:]) / self.m)
        return out
            
        

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
        self.shift = 1
        self.v_weight_arr, self.p_weight_arr = self.calc_weight_arr(self.path)
    
    def local_update(self):
        #local in time
        candidate = np.copy(self.path)
        t = np.random.choice(np.arange(self.m))
        # q = np.random.choice(np.arange(self.num_qubits))
        # candidate[t,q] = (candidate[t,q] + np.random.choice([-1,1]) + self.N) % self.N

        shift_vec = np.random.choice(np.arange(-self.shift, self.shift + 1), size=self.H.num_qubits)
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
        state = self.path[0,:]
        out = self.H.pers_curr(state)
        out = np.mean([self.H.pers_curr(self.path[i]) for i in range(self.m)])
        # out = 0 
        # for i in range(self.m):
        #     out += (self.H.pers_curr(self.path[i,:]) / self.m)
        return out



        
        

        

