#!/usr/bin/env python
# coding: utf-8

# In[35]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 
# **Upload file**

# In[36]:


data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# In[37]:


data.head()


# In[38]:


data.columns


# In[39]:


data.shape


# checking the null values

# In[41]:


data[data.isnull().any(axis = 1)]


# Quality greater than 6 is good and quality less than 6 is bad
# 1,2,3,4,5,6 - denoted by 0 which is bad
# 7,8,9 - denoted by 1 which is good

# In[44]:


clean_data = data.copy()
clean_data['Quality_label'] =(data['quality'] > 6) * 1 


# In[45]:


clean_data.head()


# **Saving Quality Label in y**

# In[46]:


y = clean_data[['Quality_label']].copy()
y.head()


# In[49]:


clean_data.columns


# Removing quality from clean_data

# In[50]:


del clean_data['quality']


# In[51]:


clean_data.columns


# Extracting the features of wine

# In[52]:


features =  ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']


# In[53]:


x = clean_data[features].copy()


# In[54]:


x.columns


# In[55]:


x.head()


# In[56]:


y.columns


# In[57]:


from sklearn.model_selection import train_test_split


# applying train test split

# In[59]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)


# In[60]:


X_train.head()


# **Applying Decision tree Classification**

# In[62]:


from sklearn.tree import DecisionTreeClassifier


# In[63]:


Wine_Quality = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
Wine_Quality.fit(X_train,y_train)


# In[64]:


type(Wine_Quality)


# In[65]:


predictions = Wine_Quality.predict(X_test)


# In[66]:


predictions[:10]


# In[67]:


y_test['Quality_label'][:10]


# In[68]:


from sklearn.metrics import accuracy_score


# Accuracy of the predicition

# In[69]:


accuracy_score(y_true = y_test, y_pred = predictions)


# In[ ]:




