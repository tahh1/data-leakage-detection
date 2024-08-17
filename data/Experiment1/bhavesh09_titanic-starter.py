#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.tree import DecisionTreeClassifier
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))
# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[3]:


submission


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train.describe(include='all')


# In[7]:


train.shape


# In[8]:


train.isnull().sum()


# In[9]:


train.info()


# In[10]:


all = pd.concat([train, test], axis=0)


# In[11]:


all.head()


# In[12]:


all.isna().sum()


# In[13]:


all.Age = all.Age.fillna(np.mean(all.Age))
all.Fare = all.Fare.fillna(np.mean(all.Fare))
all.Embarked = all.Embarked.fillna('S')


# In[14]:


all = all.drop(['Cabin','Name','Sex','Ticket','Cabin','Embarked'], axis=1)


# In[15]:


test = all[all.Survived.isnull()]
train = all[~(all.Survived.isnull())]


# In[16]:


print((train.shape, test.shape))


# In[17]:


X_train = train.drop(['Survived'],axis=1)
y_train = train.Survived
X_test = test.drop(['Survived'],axis=1)


# In[18]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[19]:


prediction = dtc.predict(X_test)
submission = pd.concat([test.PassengerId, pd.Series(prediction, index=test.index)],axis=1)
submission.columns=['PassengerId', 'Survived']
submission['Survived'] =submission['Survived'].astype(int)


# In[20]:


submission.to_csv('submission_first.csv', index=False)


# In[21]:


submission.head()


# In[ ]:




