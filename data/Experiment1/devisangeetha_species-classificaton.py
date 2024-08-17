#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")

import os
print((os.listdir("../input")))



# In[2]:


iris_data=pd.read_csv("../input/Iris.csv")
iris_data.head()


# In[3]:


iris_data.shape


# In[4]:


iris_data.info()


# 4 predictor variables/features exists in iris dataset. Species is the target variable

# In[5]:


iris_data.describe()


# In[6]:


fig, ax = plt.subplots()

sns.countplot(x='Species', data=iris_data, palette='RdBu')

plt.show()


# In[7]:


sns.pairplot(iris_data, hue="Species",palette="Set1")
plt.show()


# In[8]:


y=iris_data.Species
X=iris_data.drop('Species',axis=1).values


# In[9]:





# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21,stratify=y)


# In[10]:


knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print((knn.score(X_test, y_test)))

