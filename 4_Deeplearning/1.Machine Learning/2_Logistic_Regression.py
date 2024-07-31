# ML Project
"""
 Written by Youmin Ko 
 
 Live life as if there were no second chance
"""

############################ Logistic Regression ##############################
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import math as m

plt.close("all")

# Step 1) 1D span

Ndata = 2000
A = np.random.randn(Ndata)
f1 = plt.figure()
ax1 = plt.axes()
ax1.hist(A, bins = 10)

# Step 2) 2D span
f2 = plt.figure()
ax2 = plt.axes()
d1 = np.random.multivariate_normal(mean = [0,2],  cov = [[5,-3],[-3,6]], size = Ndata)
d2 = np.random.multivariate_normal(mean = [7,9],  cov = [[3,-1],[-1,4]], size = Ndata)
plt.scatter(d1[:,0], d1[:,1], c = 'r')
plt.scatter(d2[:,0], d2[:,1], c = 'b')

# Step 3) 3D span
f3 = plt.figure()
ax3 = plt.axes(projection = '3d')
plt.plot(d1[:,0], d1[:,1], 0 ,'r.')
plt.plot(d2[:,0], d2[:,1], 1 ,'b.')


# Step 4) X와 y 정하기
x1 = np.column_stack([np.ones([Ndata,1]), d1])
x2 = np.column_stack([np.ones([Ndata,1]), d2])
X = np.vstack([x1,x2])

y1 = np.zeros([Ndata,1])
y2 = np.ones([Ndata,1])
y = np.vstack([y1,y2])

# Step 5) Algorithm

def sigmoid(x):
    return 1/(1+np.exp(-x))

eta = 0.1
niteration = 100
wGD = np.zeros([3,1])
wGDbuffer = np.zeros([3, niteration+1])
for iteration in range(niteration):
    mu = sigmoid(wGD.T.dot(X.T).T)
    gradients = X.T.dot(mu-y)
    wGD = wGD - eta*gradients
    wGDbuffer[:, iteration+1] = [wGD[0], wGD[1], wGD[2]]

x1sig = np.linspace(-5,10,100)
x2sig = np.linspace(-5,10,100)
x1sig,x2sig = np.meshgrid(x1sig,x2sig)
ysig = sigmoid(wGD[1]*x1sig + wGD[2]*x2sig + wGD[0])
ax3.plot_surface(x1sig, x2sig, ysig, cmap = 'RdBu')
    