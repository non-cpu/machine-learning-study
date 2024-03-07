import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dfLoad = pd.read_csv('gmm/classificationSample2.txt', sep='\s+')
samples = np.array(dfLoad)
x = samples[:,0]
y = samples[:,1]
N = len(dfLoad)

plt.plot(x, y, 'b.')
# plt.show()

mx, sx = np.mean(x), np.std(x)
my, sy = np.mean(y), np.std(y)
z0 = np.array([mx+sx,my-sy]).reshape(1,2)
z1 = np.array([mx-sx,my+sy]).reshape(1,2)
Z = np.r_[z0, z1]

plt.plot(Z[:,0], Z[:,1], 'r*', markersize='16')
# plt.show()

numK = 2

iter = 0
k = np.zeros(N)

while(True):
    kOld = np.copy(k)

    for i in range(N):
        k[i] = np.linalg.norm(samples[i,:] - Z[0,:]) > np.linalg.norm(samples[i,:] - Z[1,:])
        
    if np.alltrue(kOld == k):
        break
      
    for i in range(numK):
        Z[i] = samples[k==i].mean(axis=0)

    iter += 1
 
plt.plot(Z[:,0], Z[:,1], 'g*', markersize='16')
# plt.show()

plt.close('all')

for i in range(numK):
    s = samples[k==i]
    plt.plot(s[:,0], s[:,1], '.', label=i)

plt.plot(Z[:,0], Z[:,1], 'r*', markersize='16')
plt.show()
