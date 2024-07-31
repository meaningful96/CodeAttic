# -*- coding: utf-8 -*-
"""
Written by Hanwool Jeong at Home
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd

plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/ClassificationSample2.txt", sep="\s+")

samples = np.array(dfLoad)
x = np.array(dfLoad["X"])
y = np.array(dfLoad["Y"])
N = len(x)
numK = 2

f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.plot(x, y, 'b.')

[mx, sx] = [np.mean(x), np.std(x)]
[my, sy] = [np.mean(y), np.std(y)]
z0 = np.array([mx+sx,my+sy]).reshape(1,2)
z1 = np.array([mx-sx,my-sy]).reshape(1,2)
Z = np.r_[z0, z1]
ax1.plot(Z[:,0], Z[:,1], 'r*', markersize = '20')

k = np.zeros(N)
j = 0
while(True):
    j += 1
    kOld = np.copy(k)
    for i in np.arange(N):
        z0D = np.linalg.norm(samples[i,:]-Z[0,:])
        z1D = np.linalg.norm(samples[i,:]-Z[1,:])
        k[i] = z0D > z1D
    if(np.alltrue(kOld == k)):
        break        
        
    dfCluster = pd.DataFrame(np.c_[x, y, k])
    dfCluster.columns = ["X", "Y", "K"]
    dfGroup = dfCluster.groupby("K")
    

    for cluster in range(numK):
        Z[cluster,:] = dfGroup.mean().iloc[cluster]


f2 = plt.figure(2)
ax2 = f2.add_subplot(111)
# ax2.plot(x, y, 'b.')
for (cluster, dataInCluster) in dfGroup:
    ax2.plot(dataInCluster.X, dataInCluster.Y, '.', label=cluster)
    #ax2.plot()

ax2.plot(Z[:,0], Z[:,1], 'r*', markersize = '20')
ax2.legend()

    
    
# for (cluster, dataInCluster) in dfGroup:
#     print(dataInCluster)

# N = len(x)
# np.random.seed(3)
# k = np.round(np.random.rand(N))

# npCluster = np.c_[x, y, k]
# dfCluster = pd.DataFrame(npCluster)
# dfCluster.columns = ["X", "Y", "K"]
# dfGroup = dfCluster.groupby("K")

# for (cluster, dataInCluster) in dfGroup:
#     print(dataInCluster)