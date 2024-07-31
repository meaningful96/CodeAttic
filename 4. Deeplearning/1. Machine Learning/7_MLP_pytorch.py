# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
  
#for reproducibility

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

#hyperparameters
learning_rate = 0.001 #learning rate를 parameter에 먼저 선언함*
training_epochs = 15  
batch_size = 100
drop_prob = 0.3 #dropout 확률 추가*

#MNIST dataset
#60,000개의 train data, 10,000개의 test data
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

#dataset loader
data_loader = DataLoader(dataset=mnist_train,
                                          batch_size=batch_size, #배치 크기는 100
                                          shuffle=True,
                                          drop_last=True)

###################################################### 모델 설정 ##############################################################

#MLP(Multi-Layer Perceptron)이므로 여러 Layer 설정*
linear1 = nn.Linear(784, 512, bias=True) 
linear2 = nn.Linear(512, 512, bias=True)
linear3 = nn.Linear(512, 512, bias=True)
linear4 = nn.Linear(512, 10, bias = True)

relu = nn.ReLU() #activation function으로 ReLU 설정
dropout = nn.Dropout(p=drop_prob) #dropout 설정*

#Weight를 어떤 상수로 초기화*
#Xavier uniform initialization 적용
nn.init.xavier_uniform_(linear1.weight)
nn.init.xavier_uniform_(linear2.weight)
nn.init.xavier_uniform_(linear3.weight)
nn.init.xavier_uniform_(linear4.weight)

#model 생성*
model = nn.Sequential(
        linear1, relu, dropout,
        linear2, relu, dropout,
        linear3, relu, dropout,
        linear4
        )

###################################################### Cost & Optimizer ##############################################################


#비용 함수와 optimizer 정의
criterion = nn.CrossEntropyLoss().to(device)    #내부적으로 소프트맥스 함수를 포함하고 있음
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #gradient descent할 때 adam 활용*

##################################################### Training & BackPropagation ##############################################

total_batch = len(data_loader)
model.train()    #set the model to train mode (dropout=True)*

#앞서 training_epochs의 값은 15로 지정함
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        #배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
        X = X.view(-1, 28 * 28).to(device)
        #레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y.to(device)

		#back-propagation 계산을 할 때마다 gradient 값을 누적시키기 때문에 gradient를 0으로 초기화 해주기 위한 것.
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        #gradient 계산
        cost.backward()
        #새로 계산된 w로 업데이트되고 다음 epoch로 넘어가기
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))



print('Learning finished')


# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)  #model() 사용*
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)    #model() 사용*
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()