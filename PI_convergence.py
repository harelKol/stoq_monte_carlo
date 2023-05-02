#write a code for std / mean 
import numpy as np 
from utils import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator 
from GT_current import gt_pers_curr
import matplotlib.pyplot as plt 
import pickle 

def main(px):
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

    p_ex = np.ones(len(I)) * px

    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    N = 51 #number of flux discreitization points
    x_max = 0.9 * consts.p0 #maximum flux for each squid
    b = 4
    num_samples = 24000 #30000
    first_num_mc_iter = 1 * 500000 #1 * 100000
    num_mc_iter = 2000 #2000
    m = 200 #250?
    shift = 40
    p_arr = [0.9,0.1,0]
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, p_arr)
    op_arr = sim.calc_op_arr_mp(num_samples,num_workers=8)

    with open('four_qubits_0.9_exp'+'.pickle','wb') as handle:
        pickle.dump(op_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    n_keep = 5
    gt = gt_pers_curr(I,C,L,x0,px,N,x_max,n_keep,b)
    means = np.zeros((num_samples, 4))
    stds = np.zeros((num_samples, 4))
    for i in range(1,num_samples):
        means[i,:] = np.mean(op_arr[:i],axis = 0)
        stds[i,:] = np.std(op_arr[:i],axis = 0)
    
    #plot 
    fig, axs = plt.subplots(4)
    axis = np.arange(1,num_samples)
    for i in range(4):
        err = stds[1:,i]
        axs[i].plot(axis, err)
    fig2, axs2 = plt.subplots(4)
    for i in range(4):
        curr_mean = means[1:,i] - gt[i]
        axs2[i].plot(axis, curr_mean)
    
    print(means[-1,:])
    print(gt)
    plt.show()


if __name__ == '__main__':
    main(0.9)
    
    




