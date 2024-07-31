# -*- coding: utf-8 -*-
"""
Machine learning final homework, Moon data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

plt.close("all")

[X,Y] = datasets.make_moons(n_samples = 300, shuffle = True, 
noise = 0.1, random_state = 15)

scaler = StandardScaler()
scaler.fit(X)
# standardization 수행
X_std = scaler.transform(X)

[X_train,X_test, Y_train, Y_test] = train_test_split(X_std, Y,
test_size = 0.5, random_state = 10, shuffle = True)

df_clf = pd.DataFrame(np.c_[X_std, Y])
df_clf.columns = ["x0", "x1", "object"]
df_clf_group = df_clf.groupby("object")
f3, ax3 = plt.subplots()

for label, group in df_clf_group:
    ax3.plot(group.x0, group.x1, ".", label="label")

svm_clf= SVC(C=0.1, kernel="poly", degree = 30, gamma = 0.1, coef0=50)
svm_clf.fit(X_train, Y_train) 

step = 0.1
# #  C가바뀌면 확실히 에러의 허용 범위가 바뀜. 그러면 step은? step을 바꾸니 데이터들이
# 산포되어있는 전체 그래프의 크기가 바뀜

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