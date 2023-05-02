import numpy as np 
from utils import get_hamiltonian_matrices 
from constants import consts 
import matplotlib.pyplot as plt 
import pickle 

import time 
start = time.time()

def old_H( C_inv, L_inv, Ej, x0, x_max, N):
    x = np.linspace(-x_max,x_max,N) 
    dx = 2 * x_max / (N - 1)
    p = consts.h * (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1)) / (2*dx)  #no factor of -1j to avoid complex numbers
    X1 = np.diag(x - x0[0])
    X2 = np.diag(x - x0[1])
    p2 = -(consts.h**2) * (np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1) -2*np.diag(np.ones(N),0)) / (dx**2) 
    I = np.eye(N)
    V1_harm = (L_inv[0,0] / 2) * (X1 **2)
    V2_harm = (L_inv[1,1] / 2) * (X2 **2)
    V1_unharm = Ej[0] * np.diag(np.cos(2*np.pi*x/consts.p0),0)
    V2_unharm = Ej[1] * np.diag(np.cos(2*np.pi*x/consts.p0),0)
    V1 = V1_harm + V1_unharm
    V2 = V2_harm + V2_unharm
    H1 = (C_inv[0,0] / 2) * p2 + V1
    H2 = (C_inv[1,1] / 2) * p2 + V2
    H0 = np.kron(H1,I) + np.kron(I,H2)
    V12 = L_inv[0,1] * np.kron(X1,X2)
    H = H0 - C_inv[0,1] * np.kron(p,p) + V12
    pers = L_inv[0,1] * np.kron(X1, I) + L_inv[1,1] * np.kron(I,X2)
    return H, pers


#define circuits params
I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector

#p_ex = np.array([0.6535,0.6575]) #external josephson flux, in units of flux quanta
# p_ex = np.array([1,1]) * np.pi #external josephson flux, in units of flux quanta

C = np.array([[119.5, 0],
              [0, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

L = np.array([[231.9, 0.2],
              [0.2, 238.98]]) * (1e-12) #inductance matrix, L on the diag, Mij on the off diag

x0 = np.array([0.0001, 0.0009]) * consts.p0 #external qubits flux, not in units of flux quanta
b = 4
N = 51 #number of flux discreitization points
x_max = 0.5 * consts.p0 #maximum flux for each squid

# k1 = 5
# k2 = 5
# pxs1 = np.linspace(0,0.63,k1)
# pxs2 = np.linspace(0.67,1,k2)
# pxs = np.concatenate([pxs1,pxs2])
# pxs = [0., 0.2, 0.4, 0.6, 0.663, 0.67, 0.68, 0.7, 0.73, 0.8, 0.9, 1.]
pxs = [0.67]
ops = []
for px in pxs:
    p_ex = np.array([1,1]) * px #external josephson flux, in units of flux quanta

    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const
    H_true, op = old_H(C_inv, L_inv, Ej, x0, x_max, N)
    E,V = np.linalg.eigh(H_true)
    E -= E[0]
    exp_h = np.diag(np.exp(-b*E))
    op_t = V.T @ op @ V #susspect 
    avg_op = np.trace(exp_h @ op_t) / np.trace(exp_h)
    avg_op /= (consts.mult_const)
    avg_op *= (1e6)
    ops.append(avg_op)
print(ops)
res = {'pxs':pxs,'avgs':ops}
print(res)
# plt.scatter(np.array(pxs) * np.pi,ops)
# plt.show()

