#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 08:52:54 2018

@author: wuming
"""
import pandas as pd 
train = pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/train.csv')
test = pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/test.csv')

train=train.values.tolist()
test=test.values.tolist()
b=[i[0] for i in train]
train=list(map(lambda x:x[1:],train))
train_valid=train[:8000]
b_valid=b[:8000]
train_train=train[8000:]
b_train=b[8000:]
train_train=np.array(train_train)
test=np.array(test)
b_train=np.array(b_train)



import keras
from keras.utils.np_utils import to_categorical
b= keras.utils.to_categorical(b, num_classes=10)
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(30, activation='sigmoid', input_shape=(784,)))
model.add(layers.Dense(30, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_train, b_train, epochs=5, batch_size=64)
a=model.predict(test)
a=a.tolist()
b=[a[i].index(max(a[i])) for i in range(len(a))]
num=[i+1 for i in range(len(b))]
dataframe = pd.DataFrame({'ImageId':num,'label':b})
dataframe.to_csv("/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/submit1.csv",index=False,sep=',')