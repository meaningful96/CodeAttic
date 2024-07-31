# -*- coding: utf-8 -*-
"""
Created on Sun May 23 20:36:49 2021

"""
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# plt.close("all")

[X,Y] = datasets.make_circles(n_samples = 300, shuffle = True , 
noise = 0.2, random_state = 15, factor = 0.3)
# [X,Y] = datasets.make_moons(n_samples = 300, shuffle = True, 
# noise = 0.1, random_state = 15)
# factor 안 밖의 원의 차이(작을수록 많이 차이남), noise( 노이즈 없으면 원, 있으면
# 흩뿌려진 원)
# f1, ax1 = plt.subplots()
# ax1.plot(X[:,0], X[:,1], '*')

scaler = StandardScaler()
scaler.fit(X)

# Standardization 수행
X_std = scaler.transform(X)
# f2, ax2 = plt.subplots()
# ax2.plot(X_std[:,0], X_std[:,1], '*')

[X_train,X_test, Y_train, Y_test] = train_test_split(X_std, Y,
test_size = 0.5, random_state = 10, shuffle = True)

#검증을 위해 test split 시행
df_clf = pd.DataFrame(np.c_[X_std, Y])
df_clf.columns = ["x0", "x1", "object"]
df_clf_group = df_clf.groupby("object")


f3, ax3 = plt.subplots()

for label, group in df_clf_group:
    ax3.plot(group.x0, group.x1, ".", label="label")

svm_clf = SVC(C=1, kernel="sigmoid", gamma=10)
svm_clf.fit(X_std, Y)

step = 0.01
[x0Min, x0Max] = [min(X_std[:,0])-10*step, max(X_std[:,0])+10*step]
[x1Min, x1Max] = [min(X_std[:,1])-10*step, max(X_std[:,1])+10*step]

[x0Plt, x1Plt] = np.meshgrid(np.arange(x0Min, x0Max, step), 
np.arange(x1Min, x1Max, step))
h = svm_clf.decision_function(np.c_[x0Plt.ravel(), x1Plt.ravel()])
h = h.reshape(x0Plt.shape)
CS = ax3.contour(x0Plt, x1Plt, h, cmap=plt.cm.twilight)
ax3.clabel(CS)

score = svm_clf.score(X_test, Y_test)
print(score)

#  C가바뀌면 확실히 에러의 허용 범위가 바뀜. 그러면 step은? step을 바꾸니 데이터들이
# 산포되어있는 전체 그래프의 크기가 바뀜. 