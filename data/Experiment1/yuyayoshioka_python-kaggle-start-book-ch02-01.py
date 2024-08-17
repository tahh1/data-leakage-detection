#!/usr/bin/env python
# coding: utf-8

# This notebook is a sample code with Japanese comments.
# 
# # 2.1 まずはsubmit！　順位表に載ってみよう

# In[1]:


import numpy as np
import pandas as pd


# ## データの読み込み

# In[2]:


get_ipython().system('ls ../input/titanic')


# In[3]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')


# In[4]:


gender_submission.head()


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


data = pd.concat([train, test], sort=False)


# In[8]:


data.head()


# In[9]:


print((len(train), len(test), len(data)))


# In[10]:


data.isnull().sum()


# ## 特徴量エンジニアリング

# ### 1. Pclass

# ### 2. Sex

# In[11]:


data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)


# ### 3. Embarked

# In[12]:


data['Embarked'].fillna(('S'), inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# ### 4. Fare

# In[13]:


data['Fare'].fillna(np.mean(data['Fare']), inplace=True)


# ### 5. Age

# In[14]:


age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)


# In[15]:


delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)


# In[16]:


train = data[:len(train)]
test = data[len(train):]


# In[17]:


y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
X_test = test.drop('Survived', axis=1)


# In[18]:


X_train.head()


# In[19]:


y_train.head()


# ## 機械学習アルゴリズム

# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)


# In[22]:


clf.fit(X_train, y_train)


# In[23]:


y_pred = clf.predict(X_test)


# In[24]:


y_pred[:20]


# ## 提出

# In[25]:


sub = pd.read_csv('../input/titanic/gender_submission.csv')
sub['Survived'] = list(map(int, y_pred))
sub.to_csv('submission.csv', index=False)


# In[ ]:




