#!/usr/bin/env python
# coding: utf-8

# # Extreme Gradient Boosting

# ### Importing Libraries

# In[16]:


#Importing required libraries
import pandas as pd 
import numpy as np


# ### Loading the dataset

# In[17]:


#reading the data
data=pd.read_csv('data_cleaned.csv')


# In[18]:


#first five rows of the data
data.head()


# ### Separating independent and dependent variables

# In[19]:


#independent variables
x = data.drop(['Survived'], axis=1)

#dependent variable
y = data['Survived']


# ### Creating the train and test dataset

# In[20]:


#import the train-test split
from sklearn.model_selection import train_test_split


# In[21]:


#divide into train and test sets
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 101, stratify=y)


# ## Install XGBoost

# Use the following command in terminal or command prompt
# 
# _**$ pip install xgboost**_

# ## Building an XGBM Model

# In[22]:


#Importing XGBM Classifier 
from xgboost import XGBClassifier


# In[23]:


#creating an extreme Gradient boosting instance
clf = XGBClassifier(random_state=96)


# In[24]:


#training the model
clf.fit(train_x,train_y)


# In[25]:


#calculating score on training data
clf.score(train_x, train_y)


# In[26]:


#calculating score on test data
clf.score(test_x, test_y)


# # Hyperparamter Tuning

# Same as GBDT
# 
# 1. **n_estimators:** Total number of trees
# 2. **learning_rate:**This determines the impact of each tree on the final outcome
# 3. **random_state:** The random number seed so that same random numbers are generated every time
# 4. **max_depth:** Maximum depth to which tree can grow (stopping criteria)
# 5. **subsample:** The fraction of observations to be selected for each tree. Selection is done by random sampling
# 6. **objective:** Defines Loss function (*binary:logistic* is for classification using probability, *reg:logistic* is for classification, *reg:linear* is for regression)
# 7. **colsample_bylevel:** Random feature selection at levels
# 8. **colsample_bytree:** Random feature selection at tree

# In[58]:


#set parameters
clf = XGBClassifier(base_score=0.6,learning_rate=0.5,random_state=96, colsample_bytree=0.7,n_jobs=5, max_depth=5)


# In[59]:


#training the model
clf.fit(train_x,train_y)


# In[60]:


#calculating score on test data
clf.score(test_x, test_y)


# Regularization
# 
# 1. **gamma:** Minimum reduction in loss at every split
# 2. **reg_alpha:** Makes leaf weights 0
# 3. **reg_lambda:** Decrease leaf weights more smoothly

# In[61]:


clf = XGBClassifier(gamma=0.5, random_state=15,reg_alpha=.5)


# In[62]:


#training the model
clf.fit(train_x,train_y)


# In[63]:


#calculating score on test data
clf.score(test_x, test_y)


# In[ ]:





# In[ ]:




