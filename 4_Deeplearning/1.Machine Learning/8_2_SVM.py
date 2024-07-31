# -*- coding: utf-8 -*-
"""
Written by Hanwool Jeong at Home
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.close("all") 

iris = datasets.load_iris()
X = iris["data"][0:100, (2,3)] #petal length & width
Y = iris["target"][0:100]

f1, ax1 = plt.subplots()
ax1.plot(X[:,0], X[:,1], '*')

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

f2, ax2 = plt.subplots()
ax2.plot(X_std[:,0], X_std[:,1], '*')
df_clf = pd.DataFrame(np.c_[X_std, Y])
df_clf.columns = ["petalLength", "petalWidth", "target"]
df_clf_group = df_clf.groupby("target")

f3, ax3 = plt.subplots()
for target, group in df_clf_group:
    ax3.plot(group.petalLength, group.petalWidth, '*', label = "target")
    
svm_clf = SVC(C=0.01, kernel="linear")
svm_clf.fit(X_std, Y)
#svm_clf.predict([[x1, x2]])

[x0Min, x0Max] = [min(X_std[:,0])-0.1, max(X_std[:,0])+0.1]
[x1Min, x1Max] = [min(X_std[:,1])-0.1, max(X_std[:,1])+0.1]
delta = 0.01
[x0Plt, x1Plt] = np.meshgrid(np.arange(x0Min, x0Max, delta), np.arange(x1Min, x1Max, delta))
h = svm_clf.decision_function(np.c_[x0Plt.ravel(), x1Plt.ravel()])
h = h.reshape(x0Plt.shape)
CS = ax3.contour(x0Plt, x1Plt, h, cmap=plt.cm.twilight)
ax3.clabel(CS)