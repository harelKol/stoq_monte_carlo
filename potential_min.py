from utils import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian
import numpy as np 
from scipy import optimize 
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


#define circuits params
# I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector

# #p_ex = np.array([0.6535,0.6575]) #external josephson flux, in units of flux quanta
# p_ex = np.array([1,1]) * px 
# C = np.array([[119.5, 132],
#             [132, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

# L = np.array([[231.9, 0.2],
#             [0.2, 238.98]]) * (1e-12) #inductance matrix, L on the diag, Mij on the off diag

# x0 = np.array([0.0001, 0.0009]) * consts.p0 #external qubits flux, not in units of flux quanta


p_ex = np.ones(len(I)) * px

C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

C_inv *= consts.mult_const
L_inv *= consts.mult_const
Ej *= consts.mult_const

N = 51 #number of flux discreitization points
x_max = 0.5 * consts.p0 #maximum flux for each squid
H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
x0 = np.zeros(len(I))
minimum = optimize.fmin(H.poten_min, x0)
print(H.pers_curr_min(minimum))

