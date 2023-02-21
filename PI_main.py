import numpy as np 
from squid_funcs import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator 
import matplotlib.pyplot as plt 
import pickle 

import time 
def main(px):
#define circuits params
    I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector

    #p_ex = np.array([0.6535,0.6575]) #external josephson flux, in units of flux quanta
    p_ex = np.array([1,1]) * px 
    C = np.array([[119.5, 0],
                [0, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

    L = np.array([[231.9, 0.2],
                [0.2, 238.98]]) * (1e-12) #inductance matrix, L on the diag, Mij on the off diag

    x0 = np.array([0.0001, 0.0009]) * consts.p0 #external qubits flux, not in units of flux quanta

    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    N = 51 #number of flux discreitization points
    x_max = 0.5 * consts.p0 #maximum flux for each squid
    b = 4
    num_samples = 5000
    first_num_mc_iter = 6 * 100000
    num_mc_iter = 5000

    m = 150
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N)
    sim = PI_simulator(H, m, b, num_samples, first_num_mc_iter, num_mc_iter, debug=False)
    sim.calc_op_arr()

    return sim.samples
    # print(avg_op)
    # plt.plot(sim.debug_arr)
    # plt.figure()
    # plt.hist(sim.debug_arr)
    # plt.show()
    # sim.calc_E_arr()
    # with open('PI_' + 'b_' +str(b) + '_m_' + str(m) + '_N_' + str(N) + '.pickle', 'wb') as handle:
    #     pickle.dump(sim, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('saved! m =',m)

if __name__ == '__main__':
    # pxs = [0., 0.2, 0.4, 0.6, 0.663, 0.67, 0.68, 0.7, 0.73, 0.8, 0.9, 1.]
    # avgs = []
    # for px in pxs:
    #     avgs.append(main(px))
    # res = {'pxs':pxs,'avgs':avgs}
    # with open('m_200'+'.pickle','wb') as handle:
    #     pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    samps = main(0.67)
    print('mean =',np.mean(samps))
    plt.hist(samps)
    plt.show()
    




# dx = 2 * x_max / (N - 1)
# c_max = np.trace(C_inv)
# fac = c_max * b * ((consts.h/dx)**2) / m
# print(fac)