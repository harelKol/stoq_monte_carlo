import numpy as np 
from constants import consts 
import pickle 
from multiprocessing import Pool, freeze_support
from functools import partial
from utils import get_hamiltonian_matrices
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator
import sys 
def conv_one(px, I, C, L, x0, x_max, N, m, b, num_samples, first_num_mc_iter, num_mc_iter, shift, p_arr):
    p_ex = np.ones(len(I)) * px 
    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)
    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, p_arr)
    PI_arr = sim.calc_op_arr(num_samples)
    return PI_arr
def four_qubit_conv():
    #used for specific external flux over many trials to check convergence
    trials = 8
    pxs = [0.73] * trials 

    I = np.array([3.227, 3.227, 3.227, 3.227]) * (1e-6)


    C = np.array([[119.5, 132, 0, 132],
                  [132, 120, 132, 0],
                  [0, 132, 120.5, 132],
                  [132, 0, 132, 121]]) * (1e-15)

    L = np.array([[231.9, 0.2, 0, 0.2],
                  [0.2, 232, 0.2, 0],
                  [0, 0.2, 231, 0.2],
                  [0.2, 0, 0.2, 230]]) * (1e-12) 

    x0 = np.array([0.0001, 0.0009, 0.0001, 0.0009]) * consts.p0

    N = 51 #number of flux discreitization points
    x_max = 0.73 * consts.p0 #maximum flux for each squid
    b = 4
    num_samples = 20000 #30000
    first_num_mc_iter = 5 * 100000 #1 * 100000
    num_mc_iter = 2000 #2000
    m = 150
    shift = 20
    p_arr = [0.8,0.1,0.1]

    G = partial(conv_one, I=I, C=C, L=L, x0=x0, x_max=x_max, N=N, m=m, b=b, num_samples=num_samples, 
                  first_num_mc_iter=first_num_mc_iter, num_mc_iter=num_mc_iter, shift=shift, p_arr=p_arr)
    with Pool(len(pxs)) as p:
        out = p.map(G, pxs)
    print(out)
    config = {'N':N, 'x_max':x_max, 'b':b, 'num_samples':num_samples, 'm':m, 'first_num_mc_iter':first_num_mc_iter, 
              'num_mc_iter':num_mc_iter, 'shift':shift, 'p_arr':p_arr}
    res = {'pxs':pxs, 'PI_res':out}
    dicts = [config, res] 
    with open('four_qubits_conv'+'.pickle','wb') as handle:
        pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def six_qubit_conv():
    #used for specific external flux over many trials to check convergence
    trials = 8
    pxs = [0.73] * trials 

    I = np.array([3.227, 3.227, 3.227, 3.227, 3.227, 3.227]) * (1e-6)


    C = np.array([[119.5, 130, 130, 0, 0, 130],
                [130, 120, 130, 0, 130, 0],
                [130, 130, 120.5, 130, 0, 0],
                [0, 0, 130, 121, 130, 130],
                [0, 130, 0, 130, 120.7, 130],
                [130, 0, 0, 130, 130, 119.8]]) * (1e-15)


    L = np.array([[231.9, 0.2, 0.2, 0, 0, 0.2],
                    [0.2, 232, 0.2, 0, 0.2, 0],
                    [0.2, 0.2, 231, 0.2, 0, 0],
                    [0, 0, 0.2, 230, 0.2, 0.2],
                    [0, 0.2, 0, 0.2, 230.5, 0.2],
                    [0.2, 0, 0, 0.2, 0.2, 231.3]]) * (1e-12) 

    x0 = np.array([0.0001, 0.0009, 0.0001, 0.0009, 0.0001, 0.0009]) * consts.p0

    N = 51 #number of flux discreitization points
    x_max = 0.73 * consts.p0 #maximum flux for each squid
    b = 4
    num_samples = 40000 #30000
    first_num_mc_iter = 5 * 100000 #1 * 100000
    num_mc_iter = 2000 #2000
    m = 150
    shift = 20
    p_arr = [0.8,0.1,0.1]

    G = partial(conv_one, I=I, C=C, L=L, x0=x0, x_max=x_max, N=N, m=m, b=b, num_samples=num_samples, 
                  first_num_mc_iter=first_num_mc_iter, num_mc_iter=num_mc_iter, shift=shift, p_arr=p_arr)
    with Pool(len(pxs)) as p:
        out = p.map(G, pxs)
    print(out)
    config = {'N':N, 'x_max':x_max, 'b':b, 'num_samples':num_samples, 'm':m, 'first_num_mc_iter':first_num_mc_iter, 
              'num_mc_iter':num_mc_iter, 'shift':shift, 'p_arr':p_arr}
    res = {'pxs':pxs, 'PI_res':out}
    dicts = [config, res] 
    with open('six_qubits_conv'+'.pickle','wb') as handle:
        pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    freeze_support()
    cmd = sys.argv[1]
    if cmd == '4_qubit':
        four_qubit_conv()
    elif cmd == '6_qubit':
        six_qubit_conv()
    else:
        print('invalid cmd')