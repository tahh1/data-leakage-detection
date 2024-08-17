#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# ## Importing the libraries

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Importing the dataset

# In[30]:


dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[31]:


print(x)


# ## Encoding categorical data

# In[32]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# In[33]:


print(x)


# ## Splitting the dataset into the Training set and Test set

# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ## Training the Multiple Linear Regression model on the Training set

# In[36]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# ## Predicting the Test set results

# In[37]:


y_pred =  regressor.predict(x_test)
np.set_printoptions(precision=2) # เอาแค่ ทศนิยม 2 ตำแหน่ง
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)) #1st parameter take tuple of what you want concat and 2nd need axis of show 0 mean x_axis

