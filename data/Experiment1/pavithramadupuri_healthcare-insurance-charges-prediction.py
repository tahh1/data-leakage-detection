#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


#Import packages
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("/kaggle/input/insurance/insurance.csv")


# In[4]:


df.head()


# In[5]:


#Replace string type with integer type


# In[6]:


df.sex[df.sex == 'male'] = 0
df.sex[df.sex == 'female'] = 1
df.smoker[df.smoker == 'no'] = 0
df.smoker[df.smoker == 'yes'] = 1
df.region[df.region == 'northwest'] = 0
df.region[df.region == 'southwest'] = 1
df.region[df.region == 'northeast'] = 2
df.region[df.region == 'southeast'] = 3


# In[7]:


df.head() #Make sure string types are replaced with integer type


# In[8]:


list(df.keys())


# In[9]:


print((df["charges"]))


# In[10]:


df.info()


# In[11]:


# based on above display statements, no missing values in dataset
df.shape


# In[12]:


#DATA VISUALIZATION


# In[13]:


df.plot()


# In[14]:


sns.set()


# In[15]:


df.charges.hist() # Histogram generation for "Charges" column


# In[16]:


sns.distplot(df.charges) 


# In[17]:


sns.kdeplot(df.charges) # Bell Curve generation


# In[18]:


df.hist(figsize=(12,12))
plt.plot()


# In[19]:


df.describe()


# In[20]:


#NORMALIZATION


# In[21]:


alist = [12, 33,34,23,45,21,22,36,33,32,13,43,34,24,25,26,36,35,50]
aaray = np.array(alist)
aaray


# In[22]:


plt.hist(aaray)


# In[23]:


bray = aaray/aaray.max() # it is called min-max scaler technique.
bray


# In[24]:


plt.hist(bray)


# In[25]:


Y= df[["charges"]] 
X =df.drop(columns=["charges"])
X


# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


scaler = MinMaxScaler()


# In[28]:


scaler.fit_transform(X)


# In[29]:


# Train test split
from sklearn.model_selection import train_test_split


# In[30]:


X_scaled = scaler.fit_transform(X)
X_scaled


# In[31]:


X_train,X_Test,Y_train,Y_Test = train_test_split(X_scaled,Y,test_size=0.1)


# In[32]:


X_train.shape,X_Test.shape,Y_train.shape,Y_Test.shape


# In[33]:


# machine learning
from sklearn.linear_model import LinearRegression


# In[34]:


lr = LinearRegression()


# In[35]:


lr.fit(X_train,Y_train)


# In[36]:


lr.coef_ # will give the slope


# In[37]:


lr.intercept_ # will give the constant


# In[38]:


# Testing phase
#lr.predict(X_Test)
Y_pred = lr.predict(X_Test)
Y_pred


# In[39]:


error = pd.DataFrame(Y_pred, columns=["Predicted"])
error["Actual"] = Y_Test.reset_index(drop=True)
error


# In[40]:


# Plot Actual Vs Predicted curve
plt.figure(figsize=(14, 8))
plt.plot(error.Predicted, label="Predicted")
plt.plot(error.Actual, label="Actual")
plt.legend()


# In[41]:


#Calculate the error
error["error"] = error.Predicted - error.Actual
error


# In[42]:


#Calculate Penalty
error["Penalty"] = error.error**2


# In[43]:


error.Penalty.mean()


# In[44]:


#Calculate Absolute mean error
abs(error.error).mean()


# In[ ]:




