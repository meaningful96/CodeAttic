# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('C:/Users/USER/Desktop/딥러닝/deep-learning-from-scratch-master')

################################################# Functions #############################################

import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def Sigmoid_grad(x):
    return (1.0 - Sigmoid(x)) * Sigmoid(x)
    

def ReLU(x):
    return np.maximum(0, x)


def ReLu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def SoftMax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def MSE(y, t):
    return 0.5 * np.sum((y-t)**2)


def CrossEntropyError(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = SoftMax(X)
    return CrossEntropyError(y, t)

## 제대로 동작하는지 확인 해 봄!!

if __name__ == '__main__':
    inp = np.random.rand(1,20)
    out = np.random.randn(1,20)
    
    CrossEntropyError(inp, out)
    print(CrossEntropyError(inp, out))

###################################### 수치 미분(numerical gradient) #####################################

def NumericalGradient_noBatch(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmpVal = x[idx]
        
        # f(x+h) 계산
        x[idx] = float(tmpVal) + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmpVal - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmpVal ## 값 복원
        
    return grad

def NumericalGradient(f,X):
    if X.dim == 1:
        return NumericalGradient_noBatch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = NumericalGradient_noBatch(f,x)
            
            return grad
        
####################################### 기울기를 구해보자. SimpleNet ######################################

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 class를 초기화
        
    def predict(self,x):
        return x.dot(self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = SoftMax(z)
        loss = CrossEntropyError(y, t)
        
        return loss
    
#################################### 신경망 클래스 구현하기, TwoLayerNet ##################################

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        ## 가중치, Weight 초기화
        ## weight_init_std = 가중치의 초기 표준편차이다.
        ## 평균이 0이고 표준편차가 1인 표준 정규분포에서 랜덤하게 가중치를 정하는 것이 아닌
        ## 평균이 0이고 표준편차가 0.01인 정규분포에서 가중치를 정하도록 설정하는 것이다.
        ## TwoLayerNet(784,50,10) 형태로 매개변수를 지정해야 한다.
        
        # 가중치 초기화    
        self.params = {}
        ## 일반적으로 초기 init 가중치는 정규분포를 따라 랜덤하게 선택하고 편향은 제로 벡터를 둔다.
        ## 입력층과 은닉층을 연결하는 가중치 이므로 (input_size, hidden_size) 크기의 행렬을 사용한다.
    
    
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        ## 은닉층에 사용되는 편향이므로 (hidden_size) 크기의 0으로 가득 찬 행렬을 사용한다.
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # Affine -> sigmoid -> Affine -> Softmax
        
        a1 = x.dot(W1) + b1
        z1 = Sigmoid(a1)
        a2 = z1.dot(W2) + b2
        y = SoftMax(a2)
        
        return y
    
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return CrossEntropyError(y, t)
    
    def accuracy(self, x, t):
        # argmax, argmin은 max값이나 min값이 있는 리스트의 인덱스 값을 return
        # 만약 max,min값이 여러 개면 인덱스가 낮은 걸 return
        # 만약 b = np.array([[4,3,2], [8,5,9], [7,6,1]])
        # axis = None이면 array를 일자로 편 상태에서 가장 큰 값을 반환한다.
        # 따라서 argmax(b, axis = None) = 5이다
        # 일자로 폈을때 최댓값이 9이고 그 값에 해당하는 인덱스는 5이다.
        # b.size = 3 x 3
        #        = [[4,3,2]
        #           [8,5,9]
        #           [7,6,1]]
        # axis가 1이면 (axis = 1) 각 가로축 원소들을 비교해서 최댓값의 위치를
        # array형태로 반환해줌
        # np.argmax(b,axis = 1) = array([0,2,0])
        # 1행에서 가장 큰 값 = 4(index = 0), 2행에서 가장 큰 값 = 9(index = 2)
        # axis가 0이면 (axis = 0) 각 세로축 원소들을 비교해서 return
        # 1열 max = 8(index = 1), 2열 max = 6(index = 2), 3열 max = 9(index = 1)
        
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        # axis = 1일때 즉 행에서 가장 큰 인덱스를 리턴한다.
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블  
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        # 람다는 함수를 정의해서 사용하는게 아니라 임시로 사용하는 함수
        # 즉 다른 함수들처럼 호출해서 쓸 수 있는게 아니라 
        # 저 문장 내에서만 사용할 수 있음

        ## grad는 기울기를 보관하는 딕셔너리 변수임. Numerical_gradient의 return 값        
        grads = {}
        grads['W1'] = NumericalGradient(loss_W, self.params['W1'])
        grads['b1'] = NumericalGradient(loss_W, self.params['b1'])
        grads['W2'] = NumericalGradient(loss_W, self.params['W2'])
        grads['b2'] = NumericalGradient(loss_W, self.params['b2'])
        
        # grad['W1']은 first layer의 weight의 gradient 값
        # grad['b1']은 first layer의 bias의 gradient 값
        
        return grads
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
    
        batch_num = x.shape[0]
    
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = Sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = SoftMax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = Sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads
    



##################################### MNIST 분류 진행하기 ################################################

from dataset.mnist import load_mnist

# Step 1) Load 

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

## normalize = True 는 픽셀 값을 255로 나누어 0 ~ 1 사이 값으로 나타내겠다는 뜻
## one_hot_label = True 는 0-9 정수를 one hot vector로 변환
## flatten은 default가 True임

# Step 2) Setting Hyperparameter
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

## trainSize/batchSize = 600000/100  = 100, 총 100번 반복한다는 것
## 1 에폭당 10000번, 총 1000000번

train_loss_list = []
train_acc_list = []
test_acc_list = []


# Step 3) Iteration 어떻게 뭐할 건데?

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    # 미니배치 획득
    # trainSize 이하의 정수 중 batchSize 만큼을 랜덤 추출하라는 뜻
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # Gradient 계산
    # cost fucntion의 수치미분함수에 weight와 bias를 넣어 결과를 딕셔너리로 return받음
    # grad = network.Numerical_gradient(x_batch, t_batch) -> 역전파 이용
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

### enumerate 예시 ###
Enumerate_Test = [1,5,7,33,39,52]
for i, v in enumerate(Enumerate_Test):
    print("index : {}, value: {}".format(i,v))

# index : 0, value: 1
# index : 1, value: 5
# index : 2, value: 7
# index : 3, value: 33
# index : 4, value: 39
# index : 5, value: 52