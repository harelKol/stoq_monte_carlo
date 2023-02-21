import pickle 
import matplotlib.pyplot as plt 
import numpy as np 


file_name = 'm_200'
with open(file_name + '.pickle', 'rb') as handle:
    out = pickle.load(handle)

with open(file_name + '_gt' + '.pickle', 'rb') as handle:
    out_gt = pickle.load(handle)

out_gt['avgs'] = -abs(np.array(out_gt['avgs']))
out['avgs'] = -abs(np.array(out['avgs']))

plt.scatter(out_gt['pxs'], out_gt['avgs'])
plt.scatter(out['pxs'], out['avgs'], marker='x')
plt.show()
