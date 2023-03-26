import numpy as np 
from constants import consts 
from utils import get_hamiltonian_matrices 

px = 0.9 



I = np.array([3.227, 3.227, 3.227, 3.227]) * (1e-6)


C = np.array([[119.5, 132, 0, 132],
                [132, 120, 132, 0],
                [0, 132, 120.5, 132],
                [132, 0, 132, 121]]) * (1e-15)

L = np.array([[231.9, 0.2, 0, 0.2],
                [0.2, 232, 0.2, 0],
                [0, 0.2, 231, 0.2],
                [0.2, 0, 0.2, 230]]) * (1e-12) 

x0 = np.array([0.0001, 0.0001, 0.0001, 0.0009]) * consts.p0
#x0 = np.array([0.0005, -0.0005, 0.0005, -0.0005]) * consts.p0



# I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector

# C = np.array([[119.5, 132],
#             [132, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

# L = np.array([[231.9, 0.2],
#             [0.2, 238.98]]) * (1e-12) #inductance matrix, L on the diag, Mij on the off diag

# x0 = np.array([0.0001, 0.0009]) * consts.p0 #external qubits flux, not in units of flux quanta



p_ex = np.ones(len(I)) * px 
C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)
print(C_inv)
print(L_inv)
print(Ej)
