#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection::

# Download dataset from this link:
# 
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# # Description about dataset::

# The datasets contains transactions made by credit cards in September 2013 by european cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. 
# 
# 
# ### Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# # WORKFLOW :

# 1.Load Data
# 
# 2.Check Missing Values ( If Exist ; Fill each record with mean of its feature )
# 
# 3.Standardized the Input Variables. 
# 
# 4.Split into 50% Training(Samples,Labels) , 30% Test(Samples,Labels) and 20% Validation Data(Samples,Labels).
# 
# 5.Model : input Layer (No. of features ), 3 hidden layers including 10,8,6 unit & Output Layer with activation function relu/tanh (check by experiment).
# 
# 6.Compilation Step (Note : Its a Binary problem , select loss , metrics according to it)
# 
# 7.Train the Model with Epochs (100).
# 
# 8.If the model gets overfit tune your model by changing the units , No. of layers , epochs , add dropout layer or add Regularizer according to the need .
# 
# 9.Prediction should be > 92%
# 10.Evaluation Step
# 11Prediction
# 

# # Task::

# ## Identify fraudulent credit card transactions.

# In[152]:


import pandas as pd
import numpy as np


# In[191]:


data = pd.read_csv('creditcard.csv')


# In[154]:


data.head()


# In[155]:


data.tail()


# In[156]:


data.describe()


# In[170]:


data.columns


# In[192]:


data.duplicated().sum()


# In[193]:


data.drop_duplicates(inplace=True)


# In[194]:


labels = data.iloc[:,-1]
labels


# In[195]:


data = data.iloc[:,:-1]
data.head()


# In[196]:


mean = data[['Time', 'Amount']].mean(axis=0)
data[['Time', 'Amount']] -= mean

std = data[['Time', 'Amount']].std(axis=0)
data[['Time', 'Amount']] /= std


# In[197]:


data[['Time', 'Amount']].max()


# In[175]:


mean = data.mean(axis=0)
std = data.std(axis=0)

data -= mean
data /=  std


# In[176]:


data = np.asarray(data)


# In[180]:


data.shape[0] * 20 / 100


# In[178]:


data.shape[0]


# In[181]:


85118+56745


# In[182]:


train_data = data[:85118,:]
train_labels = labels[:85118]

val_data = data[85118:141863,:]
val_labels = labels[85118:141863]

test_data = data[141863:,:]
test_labels = labels[141863:]


# In[190]:


from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(30, activation='relu', input_shape=((train_data.shape[1],))))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[188]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy'])


# In[189]:


model.fit(train_data, train_labels, epochs=5, validation_data=(val_data, val_labels), batch_size=32)


# In[119]:


model.evaluate(test_data, test_labels)


# In[145]:




