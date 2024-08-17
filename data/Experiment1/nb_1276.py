#!/usr/bin/env python
# coding: utf-8

# # Task 2 Linear Regression
# 
# ## Vishal Reddy

# ### Importing Dataset

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')



df=pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
df



# ### Splitting of data set

# In[2]:


X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X


# ### Train Test split

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# ### Training of model

# In[4]:


clf = LinearRegression().fit(X_train, y_train)


# ### Visualizing data

# In[5]:


l = clf.coef_*X+clf.intercept_
plt.scatter(X, Y)
plt.plot(X, l);
plt.show()


# ### Predicting data

# In[6]:


clf.predict(9.5)


# In[7]:


clf.decision_function(9.5)


# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:




