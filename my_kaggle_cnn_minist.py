#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 01:02:42 2018

@author: wuming
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:52:54 2018

@author: wuming
"""
import pandas as pd 
import numpy as np
from PIL import Image
import copy

train = pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/train.csv')
test = pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/test.csv')

train=train.values.tolist()
qwe=copy.deepcopy(train)
e=copy.deepcopy(train)
c=copy.deepcopy(train)
d=copy.deepcopy(train)
f=copy.deepcopy(train)
g=copy.deepcopy(train)
q=copy.deepcopy(train)
for k in range(len(train)):
    for l in range(1,len(train[0])):
        if train[k][l]!=0:
            train[k][l]+=1
a=copy.deepcopy(train)            
for k in range(len(e)):
    for l in range(1,len(e[0])):
        if e[k][l]!=0:
            e[k][l]+=1 
e=copy.deepcopy(train)            
for k in range(len(c)):
    p=c[k][0]
#    mlp=c[k]
#    del mlp[0]
    
    tem=np.array(c[k][1:])
    i=tem.reshape([28,28])
    im = Image.fromarray(i.astype('uint8'))
    im_rotate = im.rotate(10)
    matrix = np.asarray(im_rotate)
    matrix=matrix.astype('int64')
    lp=np.ravel(matrix)
    lp=list(lp)
    lp=lp[::-1]
    lp.append(p)
    lp=lp[::-1]
    d[k]=lp

for k in range(len(c)):
    p=c[k][0]
#    mlp=c[k]
#    del mlp[0]
    
    tem=np.array(c[k][1:])
    i=tem.reshape([28,28])
    im = Image.fromarray(i.astype('uint8'))
    im_rotate = im.rotate(-10)
    matrix = np.asarray(im_rotate)
    matrix=matrix.astype('int64')
    lp=np.ravel(matrix)
    lp=list(lp)
    lp=lp[::-1]
    lp.append(p)
    lp=lp[::-1]
    f[k]=lp              
            
for k in range(len(c)):
    p=c[k][0]
#    mlp=c[k]
#    del mlp[0]
    
    tem=np.array(c[k][1:])
    i=tem.reshape([28,28])
    im = Image.fromarray(i.astype('uint8'))
    im_rotate = im.rotate(-5)
    matrix = np.asarray(im_rotate)
    matrix=matrix.astype('int64')
    lp=np.ravel(matrix)
    lp=list(lp)
    lp=lp[::-1]
    lp.append(p)
    lp=lp[::-1]
    g[k]=lp                
            
for k in range(len(c)):
    p=c[k][0]
#    mlp=c[k]
#    del mlp[0]
    
    tem=np.array(c[k][1:])
    i=tem.reshape([28,28])
    im = Image.fromarray(i.astype('uint8'))
    im_rotate = im.rotate(5)
    matrix = np.asarray(im_rotate)
    matrix=matrix.astype('int64')
    lp=np.ravel(matrix)
    lp=list(lp)
    lp=lp[::-1]
    lp.append(p)
    lp=lp[::-1]
    q[k]=lp                
                        
            
            
            
            
            
            
            
            
train=a+e+d+f+qwe+g+q            
test=test.values.tolist()
#train=train[8000:]
valid=train[:8000]
b=[i[0] for i in train]
train=list(map(lambda x:x[1:],train))



train=np.array(train)
train=train.reshape(len(train),28,28,1) 
train=np.array(train)

test=np.array(test)
test=test.reshape(len(test),28,28,1) 
test=np.array(test)
b=np.array(b)
train=train.astype(np.float)
b=b.astype(np.int)











import keras
from keras.utils.np_utils import to_categorical

from keras.models import load_model
from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import SGD
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='sigmoid', input_shape=(28,28,1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3), activation='sigmoid'))
model.add(layers.MaxPooling2D((2,2)))
#model.add(layers.Dense(30, activation='relu'))
#model.add(layers.Dense(30, activation='sigmoid'))
model.add(layers.Conv2D(64,(3,3), activation='sigmoid'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10,activation='softmax'))






model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

b= keras.utils.to_categorical(b, num_classes=10)

history=model.fit(train, b,validation_split=0.20, epochs=1, batch_size=32)

 
a=model.predict(test)
a=a.tolist()
c=[a[i].index(max(a[i])) for i in range(len(a))]
num=[i+1 for i in range(len(c))]
dataframe = pd.DataFrame({'ImageId':num,'label':c})
dataframe.to_csv("/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/submit6.csv",index=False,sep=',')
        