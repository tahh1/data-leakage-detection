#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_boston
d = load_boston()


# In[3]:


import pandas as pd
data = pd.DataFrame( d.data, columns = d.feature_names)


# In[4]:


data["MEDV"] = d.target


# In[5]:


data.head()


# In[6]:


d.feature_names


# In[7]:


X = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = data[ "MEDV"]


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.3, random_state = 7)


# In[9]:


for i in [X_train, X_test, y_train, y_test]:
    print((i.head()))
    print(("_"*40))


# In[10]:


y_test.head()


# # IMPORTING LINEAR REGRESSION

# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


model = LinearRegression()


# In[13]:


import numpy as np
model.fit(X_train, y_train)


# In[14]:


#np.array(model.coef_, np.float32)


# In[15]:


actual = y_test.iloc[:10]
predicted = model.predict(X_test.iloc[:10])
model.coef_
    #isme multiple values h...kyonki humne data multiple values kya diyaa h ye logistic regression h ..linear nhi..ha


# In[16]:


for i, j in zip(actual ,predicted):
    print((i,":",j))


# # RMS VALUE

# In[17]:


from math import sqrt


# In[18]:


#apna kuch alag dhalag
sum_sq=0
for i, j in zip(actual, predicted):
    
    sum_sq+=(i-j)**2
print((sqrt(sum_sq)))


# In[19]:


actual = y_test.iloc[:]
predicted = model.predict(X_test.iloc[:])
sse = 0
for i,j in zip(actual, predicted):
    error = i-j
    sqe = error**2
    sse = sse + sqe
mse = sse/len(actual)
rmse = sqrt(mse)


# In[20]:


rmse


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


plt.scatter(X_train["CRIM"],y_train, alpha=.2)
plt.scatter(X_train["CRIM"],y_train, alpha=.2)


# In[23]:


model.score(X_test, y_test)

