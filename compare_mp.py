from GT_current import gt_pers_curr
import numpy as np 
from utils import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator 
import pickle
from multiprocessing import Pool, freeze_support
from functools import partial
import sys 

def compare_one(px, I, C, L, x0, x_max, N, m, b, num_samples, first_num_mc_iter, num_mc_iter, n_keep):
    p_ex = np.ones(len(I)) * px 
    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    sim = PI_simulator(H, m, b, num_samples, first_num_mc_iter, num_mc_iter)
    avg_PI_curr = sim.calc_op()
    avg_gt_curr = gt_pers_curr(I, C, L, x0, px, N, x_max, n_keep, b)
    return avg_PI_curr, avg_gt_curr

def compare_2_qubit():
    pxs = [0., 0.2, 0.4, 0.6, 0.663, 0.67, 0.68, 0.7, 0.73, 0.8, 0.9, 1.]
    I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector

    C = np.array([[119.5, 132],
                [132, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

    L = np.array([[231.9, 0.2],
                [0.2, 238.98]]) * (1e-12) #inductance matrix, L on the diag, Mij on the off diag

    x0 = np.array([0.0001, 0.0009]) * consts.p0 #external qubits flux, not in units of flux quanta
    N = 51 #number of flux discreitization points
    x_max = 0.5 * consts.p0 #maximum flux for each squid
    b = 4
    n_keep = 5
    num_samples = 20000 #5000
    first_num_mc_iter = 1 * 100000
    num_mc_iter = 2000 #5000
    m = 150

    G = partial(compare_one, I=I, C=C, L=L, x0=x0, x_max=x_max, N=N, m=m, b=b, num_samples=num_samples, 
                  first_num_mc_iter=first_num_mc_iter, num_mc_iter=num_mc_iter, n_keep=n_keep)
    with Pool() as p:
        out = p.map(G, pxs)
    avg_PI = [x[0] for x in out]
    avg_gt = [x[1] for x in out]
    config = {'N':N, 'x_max':x_max, 'b':b, 'num_samples':num_samples, 'm':m, 'first_num_mc_iter':first_num_mc_iter, 'num_mc_iter':num_mc_iter}
    res = {'pxs':pxs, 'gt_res':avg_gt, 'PI_res':avg_PI}
    dicts = [config, res] 
    with open('two_qubits_2'+'.pickle','wb') as handle:
        pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compare_4_qubit():
    pxs = [0., 0.2, 0.4, 0.6, 0.663, 0.67, 0.68, 0.7, 0.73, 0.8, 0.9, 1.]
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
    x_max = 0.5 * consts.p0 #maximum flux for each squid
    b = 4
    n_keep = 5
    num_samples = 20000 #5000
    first_num_mc_iter = 1 * 100000
    num_mc_iter = 2000 #5000
    m = 150

    G = partial(compare_one, I=I, C=C, L=L, x0=x0, x_max=x_max, N=N, m=m, b=b, num_samples=num_samples, 
                  first_num_mc_iter=first_num_mc_iter, num_mc_iter=num_mc_iter, n_keep=n_keep)
    with Pool() as p:
        out = p.map(G, pxs)
    avg_PI = [x[0] for x in out]
    avg_gt = [x[1] for x in out]

    config = {'N':N, 'x_max':x_max, 'b':b, 'num_samples':num_samples, 'm':m, 'first_num_mc_iter':first_num_mc_iter, 'num_mc_iter':num_mc_iter}
    res = {'pxs':pxs, 'gt_res':avg_gt, 'PI_res':avg_PI}
    dicts = [config, res] 
    with open('four_qubits_1'+'.pickle','wb') as handle:
        pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    freeze_support()
    cmd = sys.argv[1]
    if cmd == '2_qubit':
        compare_2_qubit()
        print('2 qubit')
    elif cmd == '4_qubit':
        compare_4_qubit()
    else:
        print('invalid cmd')
    





