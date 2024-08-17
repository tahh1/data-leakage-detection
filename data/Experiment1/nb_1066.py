#!/usr/bin/env python
# coding: utf-8

# In[147]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# In[148]:


dataframe = pd.read_csv("Hitters.csv")


# In[149]:


print((dataframe.head()))


# In[150]:


print((dataframe.shape))


# In[151]:


print((dataframe.isnull().sum()))


# In[156]:


dataframe.dropna(inplace = True)


# In[153]:


labels = dataframe['NewLeague']
features = dataframe.drop(['NewLeague', 'Player'], axis = 1)


# In[154]:


non_numerical = features.select_dtypes(exclude=['int64', 'float64'])                                                                                                         
numerical = features.select_dtypes(include=['int64', 'float64'])                                                                                                         
                                                                                                                                                              
non_numerical = pd.get_dummies(non_numerical)                                                                                                                                 
features = pd.concat([non_numerical, numerical], axis=1)


# In[155]:


labels = labels.replace({'A':0,'N':1})


# In[140]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[141]:


model = LogisticRegression().fit(X_train, y_train)


# In[142]:


score = model.score(X_test,y_test)


# In[146]:


print(score)


# We can compare the prediction returned by the logistic regression model with the true label by checking if the binary values match. Since both sets are binary, we can easily compare them and simply check the true positive, true negative, false positive and false negative rates. The score() function will do the work for us and telll us how accurately the model predicted values.

# In[ ]:




