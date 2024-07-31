# -*- coding: utf-8 -*-
"""
Written by Hanwool Jeong at Home
"""
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.stats

plt.close("all")

numK = 2

dfLoad = pd.read_csv('https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/ClassificationSample2.txt', sep='\s+')
samples = np.array(dfLoad)
x = samples[:,0]
y = samples[:,1]
N = len(x)

pi = np.ones(numK)*(1/numK)
mx = np.mean(x)
sx = np.std(x)
my = np.mean(y)
sy = np.std(y)

u0 = np.array([mx-sx,my+sy])
u1 = np.array([mx+sx,my-sy])
Sigma0 = np.array([[sx*sx/4, 0], [0, sy*sy/4]])
Sigma1 = np.array([[sx*sx/4, 0], [0, sy*sy/4]])

f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.plot(x,y, '.b')
ax1.plot([u0[0], u1[0]],[u0[1], u1[1]], '*r')

R = np.ones([N,numK])*(1/numK)
j = 0

while(True):
    j += 1
    N0 = sp.stats.multivariate_normal.pdf(samples, u0, Sigma0)
    N1 = sp.stats.multivariate_normal.pdf(samples, u1, Sigma1)
    
    # E-step
    Rold = np.copy(R)
    R = np.array([pi[0]*N0/(pi[0]*N0+pi[1]*N1), pi[1]*N1/(pi[0]*N0+pi[1]*N1)]).T
    
    if(np.linalg.norm(R-Rold)<N*numK*0.0001):
        break
    
    # M-step
    pi = np.ones(N).reshape(1,N).dot(R)/N    
    pi = pi.reshape(2,)
    weightedSum = samples.T.dot(R)
    
    u0 = weightedSum[:,0]/sum(R[:,0])
    u1 = weightedSum[:,1]/sum(R[:,1])
    
    Sigma0 = samples.T.dot(np.multiply(R[:,0].reshape(N,1), samples))/sum(R[:,0]) - u0.reshape(2,1)*u0.reshape(2,1).T
    Sigma1 = samples.T.dot(np.multiply(R[:,1].reshape(N,1), samples))/sum(R[:,1]) - u1.reshape(2,1)*u1.reshape(2,1).T
    
k = np.round(R[:,1])
dfCluster = pd.DataFrame(np.c_[x, y, k])
dfCluster.columns = ["X","Y","K"]
dfGroup = dfCluster.groupby("K")

f2 = plt.figure(2)
ax2 = f2.add_subplot(111)    
for (cluster, dataGroup) in dfGroup:
    ax2.plot(dataGroup.X, dataGroup.Y, ".", label = cluster)

