# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.random.randn(1000, 100) # mini batch : 1000, input : 100
node_num = 100                 # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5          # 은닉층이 5개
activations = {}               # 이곳에 활성화 결과(활성화값)를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    # Xavier 초깃값 적용          
    w = np.random.randn(node_num, node_num)*np.sqrt(2/node_num)
    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

plt.figure(figsize=(20,5))
plt.suptitle("Xavier Initialization with ReLU", fontsize=16)
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i+1) + "-layer")
    plt.ylim(0, 7000)    
    plt.hist(a.flatten(), 30, range = (0,1))


plt.show()