#animate the paths 
#define circuits params
import numpy as np 
from utils import get_hamiltonian_matrices 
from constants import consts 
from PI_hamiltonian import Hamiltonian 
from PI_sim import PI_simulator 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import pickle 

#simulate 
#2D animations of paths
def animate_v_path(V,paths):
    locs = []
    num_frames = len(paths)
    avg_path = 0
    for i in range(num_frames):
        curr_path = paths[i]
        xs = curr_path[:,0] 
        ys = curr_path[:,1] 
        locs.append((xs,ys))
        avg_path += (curr_path / num_frames)


    fig, ax = plt.subplots()
    ax.imshow(V, cmap='hot')
    line, = ax.plot(avg_path[:,0],avg_path[:,1])
    def animate(i):
        xs, ys = locs[i % num_frames]
        print(xs,ys)
        line.set_xdata(xs)
        line.set_ydata(ys)
        return line,

    ani = animation.FuncAnimation(fig, animate, interval=5, blit=True, repeat=False)
    plt.show()


def two_qubits_sim():

    I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector
    px = 0.67
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
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)
    ### TRY TO READ - ELSE SIMULATE
    try:
        with open('path_to_animate_2.pickle', 'rb') as handle:
            paths = pickle.load(handle)
    except:
        b = 4
        num_samples = 1000 #5000
        first_num_mc_iter = 1 * 100000
        num_mc_iter = 1000 #5000
        m = 150
        shift = 20
        sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, debug=False, save_paths=True)
        sim.calc_op(num_samples)
        paths = sim.debug_arr
        with open('path_to_animate_2.pickle', 'wb') as handle:
            pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #calc V 
    V = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            V[i,j] = H.poten([i,j])

    animate_v_path(V,paths)

def four_qubit_sim():
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
    p_ex = np.ones(len(I)) * px

    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    N = 51 #number of flux discreitization points
    x_max = 0.5 * consts.p0 #maximum flux for each squid
    H = Hamiltonian(C_inv, L_inv, Ej, x0, x_max, N, transform=True)

    try:
        with open('path_to_animate_4.pickle', 'rb') as handle:
            paths = pickle.load(handle)
    except:
        b = 4
        num_samples = 4000 #5000
        first_num_mc_iter = 1 * 200000
        num_mc_iter = 1000 #5000
        m = 200 #250?
        shift = 40
        
        sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, debug=False, save_paths=True)
        sim.calc_op(num_samples)
        paths = sim.debug_arr
        with open('path_to_animate_4.pickle', 'wb') as handle:
            pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ind = [0,1] #qubits to plot
    loc = np.array([int(N//2)] * len(I))
    V = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            loc[ind] = [i,j]
            V[i,j] = H.poten(loc)
    paths = [x[:,ind] for x in paths]
    
    animate_v_path(V,paths)
    
four_qubit_sim()