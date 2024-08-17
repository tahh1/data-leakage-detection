#!/usr/bin/env python
# coding: utf-8

# Fetching DataSet

# In[1]:


from sklearn .datasets import fetch_openml


# In[2]:


mnist = fetch_openml('mnist_784')


# In[3]:


mnist


# In[4]:


x, y = mnist['data'], mnist['target']


# In[5]:


x.shape


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib
import matplotlib.pyplot as plt


# In[8]:


some_digit = x[36001]
some_digit_image = some_digit.reshape(28, 28) 


# In[9]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")



# In[10]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# TO REMOVE AXIS FROM GRAPH
plt.axis("off")


# In[11]:


y[36001]


# In[12]:


x_train, x_test = x[:60000], x[60000:]


# In[13]:


y_train, y_test = y[:60000], y[60000:]


# In[14]:


import numpy as np
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# ##Creating 2 Detector

# In[15]:


y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train==2)
y_test_2 = (y_test==2)


# In[16]:


y_train_2


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[24]:


clf = LogisticRegression(tol = 0.1, solver = 'lbfgs')


# In[19]:


clf.fit(x_train, y_train_2)


# In[20]:


clf.predict([some_digit])


# In[21]:


from sklearn.model_selection import cross_val_score
cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")

