#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # Read data
# 

# In[2]:


df=pd.read_csv('data.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df['diffBreath'].value_counts()


# In[7]:


df.describe()


# In[8]:


import numpy as np


# In[9]:


def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[10]:


np.random.permutation(7)


# In[11]:


train, test=data_split(df, 0.2)


# In[12]:


train


# In[13]:


test


# In[14]:


X_train=train[['fever','Bodypain','age','runnyNose','diffBreath','cough']].to_numpy()
X_test=test[['fever','Bodypain','age','runnyNose','diffBreath','cough']].to_numpy()


# In[15]:


Y_train=train[['InfectionProb']].to_numpy().reshape(1683,)
Y_test=test[['InfectionProb']].to_numpy().reshape(420,)


# In[16]:


Y_train


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


clf=LogisticRegression()
clf.fit(X_train,Y_train)


# In[19]:


inputFeatures=[100,1,22,1,1,1]
infProb=clf.predict_proba([inputFeatures])[0][1]


# In[22]:


infProb


# In[ ]:




