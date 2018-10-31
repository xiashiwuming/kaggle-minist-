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
a=copy.deepcopy(train)
e=copy.deepcopy(train)
c=copy.deepcopy(train)
d=copy.deepcopy(train)
f=copy.deepcopy(train)
for k in range(len(train)):
    for l in range(1,len(train[0])):
        if train[k][l]!=0:
            train[k][l]+=1
for k in range(len(e)):
    for l in range(1,len(e[0])):
        if e[k][l]!=0:
            e[k][l]+=1 
            
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
            
        
            
            
            
            
            
            
            
            
            
            
train=a+e+d+f            
test=test.values.tolist()
#train=train[8000:]
valid=train[:8000]
b=[i[0] for i in train]
train=list(map(lambda x:x[1:],train))



train=np.array(train)
test=np.array(test)
b=np.array(b)
train=train.astype(np.float)
b=b.astype(np.int)



b_valid=[i[0] for i in valid]
valid=list(map(lambda x:x[1:],valid))
valid=np.array(valid)
test=np.array(test)
b_valid=np.array(b_valid)
train=train.astype(np.float)
b_valid=b_valid.astype(np.int)







import keras
from keras.utils.np_utils import to_categorical

from keras.models import load_model
from keras import models
from keras import layers
from keras import regularizers
from keras.optimizers import SGD
model = models.Sequential()
model.add(layers.Dense(512,kernel_regularizer=regularizers.l2(0.001), activation='sigmoid', input_shape=(784,)))
model.add(layers.Dropout(0.3))
#model.add(layers.Dense(30, activation='relu'))
#model.add(layers.Dense(30, activation='sigmoid'))
model.add(layers.Dense(10, activation='softmax'))



epochs = 100
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)




model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])

b= keras.utils.to_categorical(b, num_classes=10)

history=model.fit(train, b,validation_split=0.33, epochs=epochs, batch_size=32)
a=model.predict(valid)
a=a.tolist()
h=[a[i].index(max(a[i])) for i in range(len(a))]
num=[i+1 for i in range(len(b))]
p=0.0
for k in range(len(h)):
    if b_valid[k]==h[k]:
        p=p+1
m=p/len(h) 
print(m)
 
a=model.predict(test)
a=a.tolist()
b=[a[i].index(max(a[i])) for i in range(len(a))]
num=[i+1 for i in range(len(b))]
dataframe = pd.DataFrame({'ImageId':num,'label':b})
dataframe.to_csv("/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/submit6.csv",index=False,sep=',')
        