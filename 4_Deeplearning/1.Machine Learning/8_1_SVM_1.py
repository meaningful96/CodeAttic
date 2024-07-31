# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import datasets

X = np.array([[2,-3], [4,1], [0,-2], [10,3]])

scaler = StandardScaler()   # scaler라는 object를 하나 만듬 for standarlization, 데이터 타입:prprocessing._data_StandardScaler
scaler.fit(X)               # array X에 대해서 fix()라는 메서드를 사용해서 fitting함
X_std = scaler.transform(X) # fitting된 scaler를 transform을 하면, scaler.fit에서 구해진 표준편차와 평균으로 최종 연산 진행


# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import datasets

plt.close("all")

## Data Load 
iris = datasets.load_iris()

X = iris["data"][0:100, (2,3)] #pental width & length
Y = iris["target"][0:100]
## X가 데이터이고, Y는 그에 맞는 클래스임. 원래 150행인데, 마지막 50개 행은 3번째 클래스이다.
## 2개의 클래스만 가지고 함
## iris데이터느 원래 150 x 4 행렬인데, 앞의 1열과 2열은 꽃의 길이와 넓이로 명확하게 구분되기에 3열과 4열로만 분석

f1 = plt.figure(1)
ax1 = f1.add_subplot(131)
ax1.plot(X[:,0], X[:,1], '*')

## Standarlization

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

ax2 = f1.add_subplot(132)
ax2.plot(X_std[:,0], X_std[:,1], '.')

## Classification, 2개의 class로 분류

# df_clf = pd.DataFrame(np.c_[X_std, Y])
df_clf = pd.DataFrame(np.column_stack([X_std,Y]))
df_clf.columns = ["petalLength", "petalWidth", "target"]
df_clf_group = df_clf.groupby("target")

ax3 = f1.add_subplot(133)
for target, group in df_clf_group:
    ax3.plot(group.petalLength, group.petalWidth, '*', label = "target") 
    
## SVM  
    # SVC는 classification에쓰는 support vector machine이다.

from sklearn.svm import SVC    
svm_clf = SVC(C = 0.01, kernel = "linear") # C =  outlier들의 수용 정도를 나타내는 파라미터
svm_clf.fit(X_std, Y)

## 제대로 classify하는지 보기
print(svm_clf.predict([[1,1]])) # 1로 class 분류됨
print(svm_clf.predict([[0,0]])) # 0으로 class 분류됨


## 이론에서 h(x)를 보고싶으면 decision_fuction() 메서드 사용하면됨
## Contour로 나타내면 된다. Using meshgrid
## ax3.contour(x0plt, x1plt, h) h에는 svm_clf.decision_function() 넣으면됨
## 문제는 decision_function이 Design matrix의 꼴로만 받을 수 있음

[x0Min, x0Max] = [min(X_std[:,0])-0.1, max(X_std[:,0])+0.1]
[x1Min, x1Max] = [min(X_std[:,1])-0.1, max(X_std[:,1])+0.1]
delta = 0.01
[x0Plt, x1Plt] = np.meshgrid(np.arange(x0Min, x0Max, delta), np.arange(x1Min, x1Max, delta))
h = svm_clf.decision_function(np.c_[x0Plt.ravel(), x1Plt.ravel()])
h = h.reshape(x0Plt.shape)
CS = ax3.contour(x0Plt, x1Plt, h, cmap=plt.cm.twilight)
ax3.clabel(CS)


# %% contour 설명
# contour메서드를 예로들면
# ax.contour(x0plt, x1plt, h)
# x0,x1 
# [h(1,0.6), h(2,0.6), h(3,0.6)
#  h(1,0.3), h(2,0.3), h(3,0.3)] 에서 h(x0,x1)이고 이 결과과 다음과 같다 해보면
# = [1,2,1
#    1,2,2]
# h(1,0.6) = 1이 되는 것 즉, contour는 좌표에 대해서 h가 같은 것끼리 묶어버리는 것임