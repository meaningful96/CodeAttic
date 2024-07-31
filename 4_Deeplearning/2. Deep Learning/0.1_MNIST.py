# Deep Learning for AI engineer
"""
Created on Youminkk


Nil sine magno vitae labore dedit mortalibus
"""
import sys, os
sys.path.append('C:/Users/USER/Desktop/딥러닝/dataset')
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
## from 폴더.모듈  import 메서드
## load_mnist 는 불러온 데이터셋을 np.array 로 변환해주는 함수이다.

(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten = True, normalize = False)

print(x_train.shape) # (60000,784)
print(t_train.shape) # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000, ) 

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    

img = x_train[0]
label = t_train[0]
print(label)     # 5
print(img.shape) # (784, )

img = img.reshape(28,28) # 원래 이미지의 모양으로 변형
print(img.shape) # (28, 28)

img_show(img)
