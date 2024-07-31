# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

# Step 1) Data Loading
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.close("all")

dfLoad = pd.read_csv("https://raw.githubusercontent.com/meaningful96/Deep_Learning/main/1_DataSet/1_LinearRegression_dataset.txt"
                     , sep ="\s+")


xxRaw = dfLoad['xx']
yyRaw = dfLoad['yy']

plt.plot(xxRaw, yyRaw, 'ro')

# Step 2) Analytical Way, wOLS
Ndata = len(xxRaw)
xxRawNP = np.array(xxRaw)
yyRawNP = np.array(yyRaw)
X = np.column_stack([np.ones([Ndata,1]), xxRaw])

wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yyRaw)

# Step 3) Prediction
xPredict = np.linspace(0,2,101)
xPredictPadding = np.column_stack([np.ones([101,1]), xPredict])

yPredict = wOLS.T.dot(xPredictPadding.T)

# Step 4) Numerical Way, Gradient Descent
eta = 0.1
niterations = 20
wGD = np.zeros([2,1])

for iteration in range(niterations):
    gradients = -(2/Ndata)*(X.T.dot(yyRawNP.reshape(Ndata,1)-X.dot(wGD)))
    wGD = wGD - eta*gradients
    yGD = wGD.T.dot(xPredictPadding.T)
    plt.plot(xPredict.reshape(1,101), yGD, 'b*')
plt.plot(xPredict.reshape(1,101), yGD, 'g*')    
