"""
Created on meaningful96

DL Project
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close("all")

np.random.seed(0)
mu = 0.0
sigma = 1.0

x = np.linspace(-8, 8, 1000)
y = 5*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))




## min-Max Rescaling

min_val = np.min(y)
max_val = np.max(y)
x_rescale = np.linspace(0,1,1000)
rescale_minMax = (y - min_val)*((1-0)/(max_val -min_val)) + 0.

f1 = plt.figure(1)
ax1 = f1.add_subplot(131)
ax1.plot(x,y,'r-', label = "Before", alpha = 0.8)
ax1.plot(x,rescale_minMax,'b-', label = "y only", alpha = 0.8)
ax1.legend(loc='upper right')
ax1.set_title("figure 1: origin VS y-only")
plt.show()

ax2 = f1.add_subplot(132)
ax2.plot(x,rescale_minMax,'b-', label = "y only", alpha = 0.8)
ax2.plot(x_rescale,rescale_minMax,'g-', label = "x & y", alpha = 0.8)
ax2.legend(loc='upper right')
ax2.set_title("figure 2: y-only VS Final(x & y)")
plt.show()

ax3 = f1.add_subplot(133)
ax3.plot(x_rescale, rescale_minMax, 'g-', label = "x & y", alpha = 0.8)
ax3.legend(loc='upper right')
ax3.set_title("figure3: Final(x & y)")
plt.show()

### Using sklearn

from sklearn.preprocessing import MinMaxScaler

X = np.vstack([x,y]).T

for i in range(len(y)):
    # MinMaxScaler 선언 및 Fitting
    mMscaler = MinMaxScaler()
    mMscaler.fit(X)
    
    # 데이터 변환
    mMscaled_data = mMscaler.transform(X).T
  
    
f2 = plt.figure(2)

ax4 = f2.add_subplot(121)
ax4.plot(x,y,'r-', label = "Before", alpha = 0.7)
ax4.plot(mMscaled_data[0],mMscaled_data[1], "g-", label = "after")
ax4.set_title("figure 4: Orgin VS Normalization")
ax4.legend(loc='upper right')
plt.show()

ax5 = f2.add_subplot(122)
ax5.plot(mMscaled_data[0],mMscaled_data[1], "g-", label = "after")
ax5.legend(loc='upper right')
ax5.set_title("figure 5: Normalization")
plt.show()


#%%

## Standard Scailing, Standarlization

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close("all")

# 평균이 5 표준편차가 10인 가우시안 분포
np.random.seed(0)
mu = 5.0
sigma = 10.0

x = np.linspace(-25, 35, 10000)
y = 100*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

Standardization 시작
y_mean = np.mean(y)
y_var = np.var(y)
y_std = np.std(y)

y_standard = (y - y_mean)/y_std
x_standard = np.linspace(0,1,10000)

f1 = plt.figure(1)
ax1 = f1.add_subplot(131)
ax1.plot(x, y, 'r-', label = "origin", alpha = 0.8)
ax1.plot(x, y_standard, 'b-', label = "y-only", alpha = 0.8)
ax1.legend(loc="upper right")
ax1.set_title("figure 1: origin VS y-only")
plt.show()

ax2 = f1.add_subplot(132)
ax2.plot(x, y_standard, 'b-',label = "y-only", alpha = 0.8)
ax2.plot(x_standard, y_standard, 'g-', label = "Final", alpha = 0.8)
ax2.legend(loc="upper right")
ax2.set_title("figure 2: y-only VS Final(x & y)")
plt.show()

ax3 = f1.add_subplot(133)
ax3.plot(x_standard, y_standard, 'g-', label = "Final", alpha = 0.8)
ax3.legend(loc="upper right")
ax3.set_title("figure3: Final(x & y)")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.close("all")

# 평균이 5 표준편차가 10인 가우시안 분포
np.random.seed(0)
mu = 5.0
sigma = 10.0

x = np.linspace(-25, 35, 10000)
y = 100*(1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

## Using sklearn

from sklearn.preprocessing import StandardScaler

X = np.vstack([x,y]).T

# StandardScaler 선언 및 Fitting
sdscaler = StandardScaler()
sdscaler.fit(X)

# 데이터 변환
sdscaled_data = sdscaler.transform(X).T # plot하기 쉽게 Transpost한 것
""
# 데이터 프레임으로 저장
sdscaled_data = pd.DataFrame(sdscaled_data.T)

f2 = plt.figure(2)

ax4 = f2.add_subplot(121)
ax4.plot(x,y,'r-', label = "Before", alpha = 0.7)
ax4.plot(sdscaled_data[0],sdscaled_data[1], "g-", label = "after")
ax4.set_title("figure 4: Orgin VS Standardization")
ax4.legend(loc='upper right')
plt.show()

ax5 = f2.add_subplot(122)
ax5.plot(sdscaled_data[0],sdscaled_data[1], "g-", label = "after")
ax5.legend(loc='upper right')
ax5.set_title("figure 5: Standardization")
plt.show()
