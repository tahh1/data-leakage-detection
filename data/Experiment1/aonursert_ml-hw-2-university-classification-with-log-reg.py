#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Classification of Universities into two groups, Private and Public.


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("../input/us-news-and-world-reports-college-data/College.csv", index_col=0)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df["Private"] = pd.get_dummies(df["Private"], drop_first=True) # Make private column numerical


# In[7]:


df.head()


# In[8]:


# It's looking good for classification


# In[9]:


X = df.drop("Private", axis=1) # Features that I am going to use as labels
y = df["Private"] # The feature that I am going to predict


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # I split my data into train and test datas


# In[11]:


from sklearn.linear_model import LogisticRegression
logm = LogisticRegression()


# In[12]:


logm.solver = "liblinear" # Sci-kit learn changes, I have to specify
logm.fit(X_train, y_train) # I train/fit my data


# In[13]:


predictions = logm.predict(X_test)


# In[14]:


from sklearn.metrics import classification_report, confusion_matrix


# In[15]:


print((classification_report(y_test, predictions)))
print((confusion_matrix(y_test, predictions)))


# In[16]:


# This is a really good result actually.


# In[17]:


from sklearn.model_selection import cross_val_score # 10 fold cross validation scroe


# In[18]:


print((cross_val_score(logm, X, y, cv=10)))


# In[19]:


# It's a good result too

