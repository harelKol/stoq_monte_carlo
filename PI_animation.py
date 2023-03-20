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
    with open('path_to_animate.pickle', 'rb') as handle:
        paths = pickle.load(handle)
except:
    b = 4
    num_samples = 1000 #5000
    first_num_mc_iter = 1 * 100000
    num_mc_iter = 1000 #5000
    m = 150
    shift = 20
    sim = PI_simulator(H, m, b, first_num_mc_iter, num_mc_iter, shift, debug=False, save_paths=True)
    paths = sim.debug_arr
    sim.calc_op(num_samples)
    with open('path_to_animate.pickle', 'wb') as handle:
        pickle.dump(paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

#calc V 
x = np.linspace(-x_max, x_max, N) 
V = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        x1 = x[i] - x0[0]
        x2 = x[j] - x0[1]
        curr_x = np.array([x1,x2])
        V[i,j] = H.poten([i,j])


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
    line.set_xdata(xs)
    line.set_ydata(ys)
    return line,

ani = animation.FuncAnimation(fig, animate, interval=5, blit=True, repeat=False)
plt.show()
