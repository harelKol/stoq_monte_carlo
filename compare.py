from GT_current import gt_pers_curr
import numpy as np 
from utils import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator 
import pickle
pxs = [0., 0.2, 0.4, 0.6, 0.663, 0.67, 0.68, 0.7, 0.73, 0.8, 0.9, 1.]
avg_PI = []
avg_gt = []
for px in pxs:

    I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector

    p_ex = np.array([1,1]) * px 
    C = np.array([[119.5, 132],
                [132, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

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
    n_keep = 5
    num_samples = 10000 #5000
    first_num_mc_iter = 1 * 100000
    num_mc_iter = 2000 #5000
    m = 150
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    sim = PI_simulator(H, m, b, num_samples, first_num_mc_iter, num_mc_iter)
    avg_PI_curr = sim.calc_op()
    avg_gt_curr = gt_pers_curr(I, C, L, x0, px, N, x_max, n_keep, b)
    avg_PI.append(avg_PI_curr)
    avg_gt.append(avg_gt_curr)
config = {'N':N, 'x_max':x_max, 'b':b, 'num_samples':num_samples, 'm':m, 'first_num_mc_iter':first_num_mc_iter, 'num_mc_iter':num_mc_iter}
res = {'pxs':pxs, 'gt_res':avg_gt, 'PI_res':avg_PI}
dicts = [config, res] 
with open('two_qubits'+'.pickle','wb') as handle:
    pickle.dump(dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)




