import numpy as np
from utils import n_kron, get_hamiltonian_matrices
from constants import consts 

def gt_pers_curr(I, C, L, x0, px, N, x_max, n_keep, b):
    num_qubits = len(I)
    p_ex = np.ones(num_qubits) * px #external josephson flux, in units of flux quanta
    C_inv, L_inv, Ej = get_hamiltonian_matrices(I, p_ex, C, L)

    C_inv *= consts.mult_const
    L_inv *= consts.mult_const
    Ej *= consts.mult_const

    num_qubits = len(I)
    dx = 2 * x_max / (N - 1)
    H = 0
    xs_proj = []
    ps_proj = []
    #calc H projected 
    for i in range(num_qubits):
        x = np.linspace(-x_max, x_max, N) 
        p = consts.h * (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1)) / (2*dx)  #no factor of -1j to avoid complex numbers
        p2 = -(consts.h**2) * (np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1) -2*np.diag(np.ones(N),0)) / (dx**2) 

        H0 = (C_inv[i,i] / 2) * p2 + (L_inv[i,i] / 2) * np.diag((x-x0[i])**2,0) + Ej[i] * np.diag(np.cos(2*np.pi*x/consts.p0),0)
        E0,V0 = np.linalg.eigh(H0)
        ind = np.argsort(E0) #make sure the eigenvalues are sorted
        E0 = E0[ind]
        V0 = V0[:,ind]
        V0 = V0[:,:n_keep] #truncate
        E0 = E0[:n_keep]
        xs_proj.append(V0.T @ np.diag(x-x0[i]) @ V0) 
        ps_proj.append(V0.T @ p @ V0 )
        H += n_kron([np.diag(E0) if k == i else np.eye(n_keep) for k in range(num_qubits)])
    for i in range(num_qubits):
        for j in range((i+1),num_qubits):
            if C_inv[i,j] != 0:
                H += -C_inv[i,j] * n_kron([ps_proj[k] if k == i or k == j else np.eye(n_keep) for k in range(num_qubits)])
            if L_inv[i,j] != 0:
                H += L_inv[i,j] * n_kron([xs_proj[k] if k == i or k == j else np.eye(n_keep) for k in range(num_qubits)])
    full_x_ops = np.zeros((num_qubits,n_keep**num_qubits, n_keep**num_qubits))
    for i in range(num_qubits):
        full_x_ops[i,:,:] = n_kron(xs_proj[k] if k == i else np.eye(n_keep) for k in range(num_qubits))
    ops = np.einsum('ij, jkl -> ikl', L_inv, full_x_ops)
    E,V = np.linalg.eigh(H)
    E -= E[0]
    exp_h = np.diag(np.exp(-b*E))
    avg_ops = []
    for i in range(num_qubits):
        op_t = V.T @ ops[i] @ V 
        avg_op = np.trace(exp_h @ op_t) / np.trace(exp_h)
        avg_op /= (consts.mult_const)
        avg_op *= (1e6)
        avg_ops.append(avg_op)
    return avg_ops

def gt_2_qubit_example():
    I = np.array([3.227,3.157]) * (1e-6) #Josephson current vector


    C = np.array([[119.5, 132],
                  [132, 116.4]]) * (1e-15) #capacitance matrix, C on the diag, Cij on the off diag

    L = np.array([[231.9, 0.2],
                  [0.2, 238.98]]) * (1e-12) #inductance matrix, L on the diag, Mij on the off diag

    x0 = np.array([0.0001, 0.0009]) * consts.p0 #external qubits flux, not in units of flux quanta
    b = 4
    N = 51 #number of flux discreitization points
    x_max = 0.5 * consts.p0 #maximum flux for each squid
    n_keep = 10
    px = 0.67
    avg_curr = gt_pers_curr(I,C,L,x0,px, N, x_max, n_keep, b)
    print(avg_curr)

def gt_4_qubit_example():
    I = np.array([3.227, 3.227, 3.227, 3.227]) * (1e-6)

    C = np.array([[119.5, 80, 0, 80],
                  [80, 120, 80, 0],
                  [0, 80, 120.5, 80],
                  [80, 0, 80, 121]]) * (1e-15)

    L = np.array([[231.9, 0.5, 0, 0.5],
                  [0.5, 232, 0.5, 0],
                  [0, 0.5, 231, 0.5],
                  [0.5, 0, 0.5, 230]]) * (1e-12) 

    x0 = np.array([0.0005, 0.0005, 0.0005, 0.0005]) * consts.p0

    px = 0.9
    b = 4
    N = 51 #number of flux discreitization points
    x_max = 0.5 * consts.p0 #maximum flux for each squid
    n_keep = 5
    avg_curr = gt_pers_curr(I,C,L,x0,px, N, x_max, n_keep, b)
    print(avg_curr)


if __name__ == '__main__':
    gt_4_qubit_example()