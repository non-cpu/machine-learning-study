import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp

dfLoad = pd.read_csv('gmm/classificationSample2.txt', sep='\s+')
samples = np.array(dfLoad)
x = samples[:,0]
y = samples[:,1]
N = len(dfLoad)

plt.plot(x, y, 'b.')
# plt.show()

mx, sx = np.mean(x), np.std(x)
my, sy = np.mean(y), np.std(y)
u0 = np.array([mx+sx,my-sy])
u1 = np.array([mx-sx,my+sy])
Sigma0 = np.array([[sx*sx/4, 0], [0, sy*sy/4]])
Sigma1 = Sigma0.copy()

plt.plot([u0[0], u1[0]], [u0[1], u1[1]], 'r*', markersize='16')
# plt.show()

numK = 2

iter = 0
k = np.zeros(N)
R = np.ones([N,numK])*(1/numK)
pi = R.sum(axis=0)/N

while(True):
    N0 = sp.multivariate_normal.pdf(samples, u0, Sigma0)
    N1 = sp.multivariate_normal.pdf(samples, u1, Sigma1)
    
    # E-step
    Rold = np.copy(R)
    R = np.array([pi[0]*N0/(pi[0]*N0+pi[1]*N1), pi[1]*N1/(pi[0]*N0+pi[1]*N1)]).T
    
    if(np.linalg.norm(R-Rold) < N*numK*0.0001):
        break
    
    # M-step
    pi = R.sum(axis=0)/N
    weightedSum = samples.T.dot(R)
    
    u0 = weightedSum[:,0]/sum(R[:,0])
    u1 = weightedSum[:,1]/sum(R[:,1])
    
    Sigma0 = samples.T.dot(np.multiply(R[:,0].reshape(N,1), samples))/sum(R[:,0]) - u0.reshape(2,1)*u0.reshape(2,1).T
    Sigma1 = samples.T.dot(np.multiply(R[:,1].reshape(N,1), samples))/sum(R[:,1]) - u1.reshape(2,1)*u1.reshape(2,1).T
    
    iter += 1

k = np.argmax(R, axis=1)

plt.plot([u0[0], u1[0]], [u0[1], u1[1]], 'g*', markersize='16')
# plt.show()

plt.close('all')

for i in range(numK):
    s = samples[k==i]
    plt.plot(s[:,0], s[:,1], '.', label=i)

plt.plot([u0[0], u1[0]], [u0[1], u1[1]], 'r*', markersize='16')
plt.show()
