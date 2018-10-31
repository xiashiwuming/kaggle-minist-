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
train = pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/train.csv')
test = pd.read_csv('/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/test.csv')

train=train.values.tolist()
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
b= keras.utils.to_categorical(b, num_classes=10)
from keras import models
from keras import layers
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(80, activation='sigmoid', input_shape=(784,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train, b, epochs=15, batch_size=64)
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
dataframe.to_csv("/Users/wuming/study/python_study/neural-networks-and-deep-learning-master/杂例子/kaggle/minist/submit3.csv",index=False,sep=',')
        