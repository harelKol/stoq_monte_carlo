import numpy as np
from scipy import sparse 
from scipy.linalg import eigh, sqrtm
import matplotlib.pyplot as plt 
from itertools import product


h = 1.054 * 1e-34 #hbar 
p0 = 2.067 * 1e-15 #flux quanata

def n_sparse_kron(arr):
    res = 1
    for x in arr:
        res = sparse.kron(res,x)
    return res

def n_kron(arr):
    res = 1
    for x in arr:
        res = np.kron(res,x)
    return res

def n_print(arr):
    res = ""
    for x in arr:
        res += x
    return res

def qubit_basis(A):
    #pauli matrices
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
    I = np.eye(2)
    sigma = [I,X,Y,Z]
    sigma_print = ["I","X","Y","Z"]

    res = ""
    k = A.shape[0]
    n = int(np.log2(k)) #num qubits
    prod = list(product(sigma,repeat = n))
    basis = np.array([n_kron(x) for x in prod])
    print_prod = list(product(sigma_print,repeat = n))
    print_basis = np.array([n_print(x) for x in print_prod])
    for i,x in enumerate(basis):
        coff = np.trace(x @ A) / k
        coff = coff.real #should be
        if abs(coff) > 1e-2:
            curr_str = str("%.4f" % coff) + "*" + print_basis[i]
            if res != "": #not the first term
                res += (" + " + curr_str) if coff > 0 else (" - " + curr_str[1:]) 
            else:
                res += curr_str
    return res

#returns the projected Hamiltonian, and the projector in flux basis
def project(C_inv, L_inv, Ej, x0, Ns, xs_max, n_keep, get_projector = False):
    #its better not to use sparse matrices here
    num_qubits = len(Ns)
    dxs = 2 * xs_max / (Ns - 1)
    #local
    H = 0
    xs_proj = []
    ps_proj = []
    V0s = []
    for i in range(num_qubits):
        x = np.linspace(-xs_max[i],xs_max[i],Ns[i]) 
        p = h * (np.diag(np.ones(Ns[i]-1),1) - np.diag(np.ones(Ns[i]-1),-1)) / (2*dxs[i])  #no factor of -1j to avoid complex numbers
        p2 = -(h**2) * (np.diag(np.ones(Ns[i]-1),1) + np.diag(np.ones(Ns[i]-1),-1) -2*np.diag(np.ones(Ns[i]),0)) / (dxs[i]**2) 
        H0 = (C_inv[i,i] / 2) * p2 + (L_inv[i,i] / 2) * np.diag((x-x0[i])**2,0) + Ej[i] * np.diag(np.cos(2*np.pi*x/p0),0)
        E0,V0 = np.linalg.eigh(H0)
        ind = np.argsort(E0) #make sure the eigenvalues are sorted
        E0 = E0[ind]
        V0 = V0[:,ind]
        V0 = V0[:,:n_keep[i]] #truncate
        if get_projector:
            V0s.append(V0)
        E0 = E0[:n_keep[i]]
        xs_proj.append(V0.T @ np.diag(x-x0[i]) @ V0) 
        ps_proj.append(V0.T @ p @ V0 )
        H += n_kron([np.diag(E0) if k == i else np.eye(n_keep[k]) for k in range(num_qubits)])
    for i in range(num_qubits):
        for j in range((i+1),num_qubits):
            if C_inv[i,j] != 0:
                H += -C_inv[i,j] * n_kron([ps_proj[k] if k == i or k == j else np.eye(n_keep[k]) for k in range(num_qubits)])
            if L_inv[i,j] != 0:
                H += L_inv[i,j] * n_kron([xs_proj[k] if k == i or k == j else np.eye(n_keep[k]) for k in range(num_qubits)])
    return H, V0s

def project_diag(C_inv, L_inv, Ej, x0, Ns, xs_max, n_keep, eig_vec_in_x = False):
    num_qubits = len(Ns)
    H, V0s = project(C_inv, L_inv, Ej, x0, Ns, xs_max, n_keep, get_projector = eig_vec_in_x)
    E,V = eigh(H,subset_by_index=[0,(2**num_qubits - 1)])
    if eig_vec_in_x:
        V = n_kron(V0s) @ V
    E = E / (2 * np.pi * h * 1e9) #Ghz
    return E,V

def get_hamiltonian_matrices(I, p_ex, C, L):
    N = C.shape[0] #qubits num
    L_inv = np.zeros((N,N))
    Ej = np.zeros(N)
    for i in range(N):
        C[i,i] = np.sum(C[i,:]) #capacitance loading
        L_inv[i,i] = 1 / L[i,i]
        Ej[i] = -p0 * I[i] * np.cos(np.pi * p_ex[i]) / (2*np.pi)
        for j in range(N):
            if j != i:
                L_inv[i,j] = L[i,j] / (L[i,i] * L[j,j])
                C[i,j] = -C[i,j]
    C_inv = np.linalg.inv(C)
    return C_inv, L_inv, Ej

def first_order_SW(C_inv, L_inv, Ej, x0, Ns, xs_max, get_pauli_string = True):
    num_qubits = len(Ns)
    H, _ = project(C_inv, L_inv, Ej, x0, Ns, xs_max, 2*np.ones(num_qubits).astype(int))
    H = H / (2 * np.pi * h * 1e9)
    pauli_str = qubit_basis(H) if get_pauli_string else ''
    return H, pauli_str

def full_SW(C_inv, L_inv, Ej, x0, Ns, xs_max, n_keep, get_pauli_string = True):
    num_qubits = len(Ns)
    H, _ = project(C_inv, L_inv, Ej, x0, Ns, xs_max, n_keep)
    E, V = eigh(H,subset_by_index=[0,(2**num_qubits) - 1])
    e_ind = np.argsort(E)
    V = V[:,e_ind]
    num_states = V.shape[1]
    num_basis = V.shape[0]
    P = sum([np.outer(V[:,i],V[:,i]) for i in range(num_states)])
    P0_arr = []
    for i in range(num_qubits):
        P0_curr = np.zeros((n_keep[i],n_keep[i]))
        P0_curr[0,0] = 1
        P0_curr[1,1] = 1
        P0_arr.append(P0_curr)
    P0 = n_kron(P0_arr)
    Rp = 2*P - np.eye(num_basis)
    Rp0 = 2*P0 - np.eye(num_basis)
    U = sqrtm(Rp0 @ Rp)
    U = U.real #imaginary part is an error
    H_q = U @ H @ U.T 
    ind = np.where(np.diag(P0) == 0)[0]
    H_q = np.delete(H_q, ind, axis = 0)
    H_q = np.delete(H_q, ind, axis = 1)
    H_q = H_q / (2 * np.pi * h * 1e9)
    pauli_str = qubit_basis(H_q) if get_pauli_string else ''
    return H_q, pauli_str
    
def debug_poten(L, Ej, N, x0, x_max):
    X = np.linspace(-x_max, x_max, N)
    V = ((X - x0)**2) / (2*L) + Ej * np.cos(2*np.pi*X/p0)
    plt.plot(X, V)
    plt.show()

def debug_qubit(L, C, Ej, N, x0, x_max):
    dx = 2 * x_max / (N - 1)
    X = np.linspace(-x_max, x_max, N)
    V = np.diag(((X - x0)**2) / (2*L) + Ej * np.cos(2*np.pi*X/p0))
    p2 = -(h**2) * (np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1) -2*np.diag(np.ones(N),0)) / (dx**2)
    H = (p2 / (2 * C)) + V 
    E, V = eigh(H)
    E = (E[0:3] - E[0]) / (2 * np.pi * h * 1e9)
    print(E)
    for e in E:
        plt.plot(np.arange(10), np.ones(10)*e)
    plt.show()

def design(H):
    N = H.shape[0]
    H_d = np.zeros_like(H)
    for i in range(N):
        for j in range(N):
            if i == j:
                H_d[i,j] = H[i,j]
            else:
                H_d[i,j] = - abs(H[i,j])
    return H_d 





