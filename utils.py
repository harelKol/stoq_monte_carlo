import numpy as np
from constants import consts
def n_kron(arr):
    res = 1
    for x in arr:
        res = np.kron(res,x)
    return res

def get_hamiltonian_matrices(I, p_ex, C, L):
    C_here = np.copy(C)
    N = C_here.shape[0] #qubits num
    L_inv = np.zeros((N,N))
    Ej = np.zeros(N)
    for i in range(N):
        C_here[i,i] = np.sum(C_here[i,:]) #capacitance loading
        L_inv[i,i] = 1 / L[i,i]
        Ej[i] = -consts.p0 * I[i] * np.cos(np.pi * p_ex[i]) / (2*np.pi)
        for j in range(N):
            if j != i:
                L_inv[i,j] = L[i,j] / (L[i,i] * L[j,j])
                C_here[i,j] = -C_here[i,j]
    C_inv = np.linalg.inv(C_here)
    return C_inv, L_inv, Ej

#add normal mode transformation