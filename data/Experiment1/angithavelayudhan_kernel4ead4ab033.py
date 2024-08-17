#!/usr/bin/env python
# coding: utf-8

# importing library
# *  NumPy is a package in Python used for Scientific Computing. The ndarray (NumPy Array) is a multidimensional array used to store values of same datatype
# * matplotlib.pyplot for data visualization 
# * pandas for data manipulation  
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing dataset to x and y

# In[2]:


dataset = pd.read_csv('../input/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
# * note: early its was in sklearn.cross_validation for new version it will be in sklearn.model_selection

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Feature Scaling no need here
# * """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""

# * training aimple linear Regression model

# In[4]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# testing model

# In[5]:


y_pred=regressor.predict(X_test)
print(y_pred)


# data visualization of training set

# In[6]:


plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='b')
plt.title('salary vs experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()


# data visualization of test set

# In[7]:


plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='b')
plt.title('salary vs experience(test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

