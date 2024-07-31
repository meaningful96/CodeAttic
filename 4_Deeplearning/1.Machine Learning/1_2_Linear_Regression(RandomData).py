# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################# Linear Regression #################################################


## Step 1) Random data 생성
plt.close('all')

n = 200
np.random.seed(1)
A = np.random.randn(n).reshape(100,2)

f1 = plt.figure()
ax1 = plt.axes()
ax1.plot(A[:,0], A[:,1], 'r.')


## Step 2) wOLS, Analytical way
x = np.array(A[:,0])
y = np.array(A[:,1])
X = np.column_stack([np.ones([len(x),1]), x])

wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y).reshape(2,1)

## Step 3) Prediction Data 만들기
xPrediction = np.linspace(-3, 3, 100) # np.max(x) = 2.528, np.mean(y) = -1.857
xPredictionPadding = np.column_stack([np.ones([100,1]), xPrediction])
yPrediction = wOLS.T.dot(xPredictionPadding.T)

## Step 4) Gradient Descent, Numerical way
eta = 0.3 # learning rate
wGD = np.zeros([2,1])
niteration = 100
for i in range(niteration):
    gradients = (-2/len(x))*(X.T.dot(yPrediction.reshape(100,1) - X.dot(wGD)))
    wGD = wGD - eta*gradients
    yGD = wGD.T.dot(xPredictionPadding.T)
    plt.plot(xPrediction.reshape(1,100), yGD, 'b.')
    
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(xPrediction.reshape(1,100), yGD, 'g.')
plt.legend()
