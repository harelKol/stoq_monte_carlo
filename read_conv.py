import numpy as np 
import matplotlib.pyplot as plt 
from GT_current import gt_pers_curr 
from constants import consts 
import pickle 
file_name = 'four_qubits_0.9_exp'
with open(file_name + '.pickle', 'rb') as handle:
    op_arr = pickle.load(handle)

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

#x0 = np.array([0.0001, 0.0001, 0.0001, 0.0009]) * consts.p0
x0 = np.array([0.0005, -0.0005, 0.0005, -0.0005]) * consts.p0

p_ex = np.ones(len(I)) * px

n_keep = 5
N = 51 #number of flux discreitization points
x_max = 0.5 * consts.p0 #maximum flux for each squid
b = 4
num_samples = len(op_arr)
gt = gt_pers_curr(I,C,L,x0,px,N,x_max,n_keep,b)
means = np.zeros((num_samples, 4))
stds = np.zeros((num_samples, 4))
for i in range(1,num_samples):
    means[i,:] = np.mean(op_arr[:i],axis = 0)
    stds[i,:] = np.std(op_arr[:i],axis = 0)

#plot 
fig1, axs1 = plt.subplots(4)
axis = np.arange(1,num_samples)
for i in range(4):
    err = stds[1:,i]
    axs1[i].plot(axis, err)
fig2, axs2 = plt.subplots(4)
for i in range(4):
    curr_mean = means[1:,i] - gt[i]
    axs2[i].plot(axis, curr_mean)

fig3, axs3 = plt.subplots(4)
for i in range(4):
    axs3[i].plot(op_arr[:,i])


print(means[-1,:])
print(gt)

fig1.savefig('try1')
fig2.savefig('try2')
fig3.savefig('try3.png')