# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""


import sys
import os
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist

## Step 1) Data Load
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test



## Step 2) 함수 정의

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x
    return y

def ReLU(x):
    return np.maximum(0,x)
## Step 3) 네트워크 구성

def init_network():
    with open("C:/Users/USER/Desktop/딥러닝/deep-learning-from-scratch-master/ch03/sample_weight.pkl", 'rb') as f:
        # 학습된 가중치 매개변수가 담긴 파일
        # 학습 없이 바로 추론을 수행
        network = pickle.load(f)
    return network     

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = x.dot(W1) + b1
    z1 = sigmoid(a1)
    a2 = z1.dot(W2) + b2
    z2 = sigmoid(a2)
    a3 = z2.dot(W3) + b3
    y = softmax(a3)
    return y

## Step 4) 데이터 입력

x, t = get_data()
network = init_network()
accuracy_cnt = 0


## Step 5) 배치화 하기
batch_size = 100
for i in range(0,len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) 
    
print("Accuracy: " + str(float(accuracy_cnt/len(x))))  