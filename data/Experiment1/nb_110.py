#!/usr/bin/env python
# coding: utf-8

# In[20]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# In[21]:


iris = load_iris()


# In[22]:


print((iris.data.T))


# In[39]:


print(iris)


# In[41]:


type(iris.data)


# In[42]:


type(iris)


# In[45]:


features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_Length_Label = iris.feature_names[0]
sepal_Width_Label = iris.feature_names[1]
petal_length_Label = iris.feature_names[2]
petal_width_Label = iris.feature_names[3]

for x in range(4):
    print((iris.feature_names[x]))


# In[25]:


plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 50

plt.scatter(sepal_length, sepal_width, s=30, c=iris.target)
plt.xlabel(sepal_Length_Label)
plt.ylabel(sepal_Width_Label)
plt.show()


# In[26]:


df = pd.DataFrame(iris.data)


# In[27]:


df


# In[28]:


corr = df.corr()
corr.style.background_gradient(cmap='viridis')


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


# In[36]:


X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
print((x_new.shape))


# In[37]:


prediction = knn.predict(X_new)
print(prediction)


# In[38]:


print((knn.score(X_test, y_test)))


# In[ ]:




