#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#importing the dataset
dataFrame = pd.read_csv("Salary_Data.csv")


# In[4]:


#Checking the dataset for missing values
dataFrame.isnull().sum()


# In[5]:


dataFrame.describe()


# In[6]:


dataFrame.info()


# In[7]:


#Visualising the dataset
plt.scatter(x="YearsExperience",y="Salary",data=dataFrame)


# In[8]:


#Getting data
x = dataFrame[["YearsExperience"]]
y = dataFrame[["Salary"]]


# In[9]:


#Splitting the dataset as Train and Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size = 0.3, random_state = 11)


# In[10]:


#Model building
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(x_train,y_train)


# In[11]:


#Coefficient
model.coef_


# In[12]:


#Intercept
model.intercept_


# In[13]:


# R-square
model.score(x_train,y_train)


# In[14]:


pred = lr.predict(x)


# In[15]:


pred = pd.DataFrame(pred)


# In[16]:


#Sorting by index
x_train = x_train.sort_index()
y_train = y_train.sort_index()
pred = pred.sort_index()


# In[17]:


#Visualising the model
plt.scatter(x="YearsExperience",y="Salary",data=dataFrame)
plt.plot(x,pred,"r")
plt.title("SimpleLinearRegression")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")

