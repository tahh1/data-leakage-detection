#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# ## In this kernel, XGBoost algorithm has been explained. 

# In[2]:


data = pd.read_csv('../input/Breast_cancer_data.csv')
data.head(10)


# In[3]:


X = data.iloc[:,0:5].values
y = data.iloc[:,5].values


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# # **XGBoost Algorithm**

# * XGBoost stands for e**X**treme **G**radient **B**oosting.
# * XGBoost is an algorithm that has recently been dominating applied machine learning and competitions for structured or tabular data.
# * XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
# * XGBoost supports the following main interfaces: CLI, C++, Python, R, Julia, Java...
# * The implementation of the algorithm was engineered for efficiency of compute time and memory resources. A design goal was to make the best use of available resources to train the model. 
# * The two reasons to use XGBoost are also the two goals of the project:
#        1. Execution Speed.
#        2. Model Performance.

# In[5]:


from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# In[6]:


y_pred = classifier.predict(X_test)
y_pred


# In[7]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
cm


# # Conclusion

# Compared with the result of artificial neural network (ann) algorithm, XGBoost is a little bit more successful. You can find my ANN kernel link below.
# 
# https://www.kaggle.com/armagansarikey/prediction-with-artificial-neural-network-keras
# 
# My other related kernels:
# 
# https://www.kaggle.com/armagansarikey/how-to-select-model-k-fold-cv-gridsearch
# 
# https://www.kaggle.com/armagansarikey/machine-learning-1-data-preprocessing
# 
# If you have any question or suggest, I will be happy to hear it.
# 
# If you like it, please upvote :)
# 
