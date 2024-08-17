#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics 


# In[14]:


df = pd.read_csv('http://bit.ly/w-data')


# In[15]:


df.head()


# In[16]:


df.info()


# In[17]:


df.shape


# In[18]:


df.plot(x="Hours", y="Scores", style= '.')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[19]:


X = df['Hours'].values.reshape(-1,1)
y = df['Scores'].values


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[21]:


model = LinearRegression(normalize=True)
model.fit(X_train, y_train) 
print("Model Trained")


# In[22]:


line = model.coef_*X+model.intercept_
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[23]:


print(('Test Score', model.score(X_test, y_test)))
print(('Training Score', model.score(X_train, y_train)))


# In[24]:


print(X_test)
y_pred = model.predict(X_test)
print(y_pred)


# In[25]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[26]:


print(('Score of the student hours = 9.5', model.predict([[9.25]])))


# In[27]:


print(('Mean Absolute Error', metrics.mean_absolute_error(y_test, y_pred)))
print(('Mean squared Error', metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[ ]:




