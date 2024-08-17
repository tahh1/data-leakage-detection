#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


import sklearn.datasets as dt


# In[3]:


dic = dt.load_digits()
list(dic.keys())


# In[4]:


dic.data.shape


# In[5]:


dic.images.shape


# In[6]:


import matplotlib.pyplot as plt
plt.imshow(dic.images[200])


# In[7]:


X = dic.data
y = dic.target


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[9]:


y_train.shape
X_train.shape


# In[10]:


y_test.shape
y_train.shape


# In[11]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn
model = knn.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_score = model.score(X_test, y_test)
y_pred
y_score


# In[12]:


import sklearn.metrics
model = str(round(model.score(X_test,y_test) * 100, 2))+"%"
print(("O modelo k-NN foi",model))

