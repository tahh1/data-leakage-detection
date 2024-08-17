#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


from sklearn.datasets import load_wine


# In[3]:


wine_data = load_wine()


# In[4]:


data = pd.DataFrame(wine_data["data"], columns=wine_data["feature_names"])
data["Target"] = wine_data["target"]


# In[5]:


data.head()


# In[6]:


from sklearn.tree import DecisionTreeClassifier


# In[7]:


x = data.drop("Target", axis=1)
y = data["Target"]


# In[8]:


clf = DecisionTreeClassifier(random_state=42)


# In[9]:


clf.fit(x,y)


# In[10]:


print((clf.predict([x.loc[90]])))
print((data.loc[90]))


# In[11]:


from sklearn.model_selection import cross_val_score


# In[12]:


clf = DecisionTreeClassifier(random_state=42, max_depth=3, max_features=3)
cross_val = cross_val_score(clf, x,y, cv=5)
print((cross_val.mean()))


# In[13]:


print(cross_val)


# In[14]:


from sklearn.preprocessing import StandardScaler


# In[15]:


skl = StandardScaler()
skl.fit(x)
x_scaled = skl.transform(x)


# In[16]:


clf = DecisionTreeClassifier(random_state=42)
model = cross_val_score(clf, x, y, cv=5)
print((model.mean()))
model = cross_val_score(clf, x_scaled, y, cv=5)
print((model.mean()))


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


# In[19]:


score = []
for i in range(1,80):
    knn = KNeighborsClassifier(n_neighbors=i)
    cv = KFold(random_state=42, n_splits=5, shuffle=True)
    res = cross_val_score(knn, x_scaled, y, cv=cv)
    score.append(res.mean())


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


plt.plot(score);
plt.grid()
print((max(score)))
print((score.index(max(score))+1))


# In[22]:


knn = KNeighborsClassifier(n_neighbors=29)
res = cross_val_score(knn, x_scaled, y, cv=5)
print((res.mean()))


# In[23]:


knn.fit(x_scaled, y);
knn.predict([x_scaled[98]])

