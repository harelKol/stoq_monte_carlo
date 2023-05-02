import numpy as np 
from utils import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator 
from GT_current import gt_pers_curr

def main_four(px):
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
    #x0 = np.array([0.0005, -0.0005, 0.0005, -0.0005]) * consts.p0

    p_ex = np.ones(len(I)) * px

    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    N = 51 #number of flux discreitization points
    x_max = 0.73 * consts.p0 #maximum flux for each squid
    b = 4
    n_keep = 5 
    num_samples = 40000 #30000
    first_num_mc_iter = 10 * 100000 #1 * 100000
    num_mc_iter = 2000 #2000
    m = 150
    shift = 20
    p_arr = [0.8,0.1,0.1]

    avg_gt = gt_pers_curr(I, C, L, x0, px, N, x_max, n_keep, b)
    print('GT =', avg_gt) 

    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, p_arr)
    # avg_curr = sim.calc_op(num_samples)
    avg_curr = sim.calc_op_mp(num_samples,num_workers=8)
    print('PI =',avg_curr)    

def main_six(px):
    I = np.array([3.227, 3.227, 3.227, 3.227, 3.227, 3.227]) * (1e-6)

    C_cp = 130
    C = np.array([[119.5, C_cp, 0, 0, 0, C_cp],
                    [C_cp, 120, C_cp, 0, 0, 0],
                    [0, C_cp, 120.5, C_cp, 0, 0],
                    [0, 0, C_cp, 121, C_cp, 0],
                    [0, 0, 0, C_cp, 120.7, C_cp],
                    [C_cp, 0, 0, 0, C_cp, 119.8]]) * (1e-15)


    L = np.array([[231.9, 0.2, 0, 0, 0, 0.2],
                    [0.2, 232, 0.2, 0, 0, 0],
                    [0, 0.2, 231, 0.2, 0, 0],
                    [0, 0, 0.2, 230, 0.2, 0],
                    [0, 0, 0, 0.2, 230.5, 0.2],
                    [0.2, 0, 0, 0, 0.2, 231.3]]) * (1e-12) 


    x0 = np.array([0.0001, 0.0009, 0.0001, 0.0009, 0.0001, 0.0009]) * consts.p0

    p_ex = np.ones(len(I)) * px


    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    N = 51 #number of flux discreitization points
    x_max = 0.73 * consts.p0 #maximum flux for each squid
    b = 4
    n_keep = 5 
    num_samples = 40000 #30000
    first_num_mc_iter = 5 * 1000000 
    num_mc_iter = 2000 #2000
    m = 150
    shift = 20 #20 sec 30 third
    p_arr = [0.6, 0.3, 0.1]

    # avg_gt = gt_pers_curr(I, C, L, x0, px, N, x_max, n_keep, b)
    # print('GT =', avg_gt) 

    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, p_arr)
    avg_curr = sim.calc_op_mp(num_samples,num_workers=8)
    print('PI =',avg_curr)
if __name__ == '__main__':
    main_six(0.73)
    
    




