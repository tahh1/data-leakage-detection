#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("../input/insurance.csv")


# In[4]:


df.info()


# In[5]:


df.columns
df.head()


# In[6]:


#df.smoker[df.smoker == 'yes']=1
#df.smoker[df.smoker == 'no']=0
#df=df.drop(["sex","region"],axis=1)
df1_cast= df.infer_objects()


# In[7]:


df1_cast.head()


# In[8]:


df1_cast.corr()


# In[9]:


#change all the string to one hot encoding
df_onehotencode =pd.get_dummies(df1_cast)


# In[10]:


df_onehotencode.head()


# In[11]:


df_onehotencode.info()


# In[12]:


df_onehotencode.corr()
#expenses and smoker is +- 78% correlated


# In[ ]:





# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[14]:


X= df_onehotencode.drop("expenses",axis=1)
Y = df_onehotencode["expenses"]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)


# In[16]:


X_train.shape


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(X_train,y_train)


# In[19]:


model.intercept_


# In[20]:


model.coef_


# In[21]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


# In[22]:


train_predict = model.predict(X_train)


# In[23]:


test_predict = model.predict(X_test)


# In[24]:


print(("Train:",mean_absolute_error(y_train,train_predict)))


# In[25]:


print(("Test:",mean_absolute_error(y_test,test_predict)))


# In[26]:


print(("Train:",mean_squared_error(y_train,train_predict)))


# In[27]:


print(("Test:",mean_squared_error(y_test,test_predict)))


# In[28]:


print(('RMSE train',np.sqrt(mean_squared_error(y_train,train_predict))))
print(('RMSE test',np.sqrt(mean_squared_error(y_test,test_predict))))


# In[29]:


print(('r2 train',r2_score(y_train,train_predict)))
print(('r2 test',r2_score(y_test,test_predict)))


# In[30]:


import seaborn as sns


# In[31]:


for i in df_onehotencode.columns:
        sns.pairplot(data=df_onehotencode,x_vars=i,y_vars='expenses')


# In[ ]:





# In[ ]:




