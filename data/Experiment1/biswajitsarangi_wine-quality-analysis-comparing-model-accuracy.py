#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# - Loading the data and performing data analysis, as the data is quiute uniform in ways like white wines are listed first followed by red wines.
# - We'll also look into various parameters and their behaviour, on the dataset

# In[2]:


dataset = pd.read_csv("/kaggle/input/wine-quality/winequalityN.csv")


# In[3]:


dataset.head()  # looking into initial 5 rows of dataset


# In[4]:


dataset.tail()  # looking into last 5 rows of dataset


# In[5]:


dataset.describe()


# Before processing any dataset it important to replace all the null values for computation flexibility, and replace their values with the mean value of that particular row.

# In[6]:


dataset.isnull().any()  


# In[7]:


dataset['fixed acidity'].fillna(dataset['fixed acidity'].mean(),inplace = True)
dataset['volatile acidity'].fillna(dataset['volatile acidity'].mean(),inplace = True)
dataset['citric acid'].fillna(dataset['citric acid'].mean(),inplace = True)
dataset['residual sugar'].fillna(dataset['residual sugar'].mean(),inplace = True)
dataset['chlorides'].fillna(dataset['chlorides'].mean(),inplace = True)
dataset['pH'].fillna(dataset['pH'].mean(),inplace = True)
dataset['sulphates'].fillna(dataset['sulphates'].mean(),inplace = True)


# In[8]:


dataset.isnull().any()  # Now we have removed all missing or null values and replaced them by the mean.


# Now, let's look into how each feature affects the quality of the Wine by plotting various graphs and histograms.

# In[9]:


dataset.corr()


# In[10]:


import seaborn as sn

corrmat = dataset.corr()
sn.heatmap(corrmat,annot = True)


# In[11]:


dataset.quality.hist(bins=10)


# In[12]:


# Now we will se the variations of each factor against the overall quality,

columns = list(dataset.columns)
columns.remove('type')
columns.remove('quality')


for i in columns:
  fig = plt.figure(figsize = (10,6))
  sn.barplot(x = 'quality', y = i, data = dataset)


# Since the data is in ordered format we will shuffle the dataframe before dividing into train and test.

# In[13]:


dataset = dataset.sample(frac=1).reset_index(drop=True)


# In[14]:


dataset.head()


# In[15]:


dataset.tail()


# In[16]:


for i in range(len(dataset['quality'])):
    if dataset['quality'][i] <= 6.5:
        dataset['quality'][i] = 0
    else:
        dataset['quality'][i] = 1
        
# We classify the good and bad wine with, using a certain threshold.


# In[17]:


lb = LabelEncoder()

dataset['type'] = lb.fit_transform(dataset['type'])


# # Training and Testing
# 
# We divide the dataset into dependent and independent variables, and convert them into numpy arrays.
# Also, as we can see we have large differences in the values hence we scale all the values inside columns, so that its distribution will have a mean value 0 and standard deviation of 1.

# In[18]:


x = dataset.iloc[:,0:12].values
y = dataset.iloc[:,12:].values


# In[19]:


x # we will scale these values


# In[20]:


y


# In[21]:


sc = StandardScaler()


# In[22]:


x = sc.fit_transform(x)
x


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


# In[24]:


x_train.shape


# In[25]:


x_test.shape


# In[26]:


y_train.shape


# In[27]:


y_test.shape


# In[28]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# # Random Forest

# In[29]:


rfc = RandomForestClassifier(n_estimators=400)
rfc.fit(x_train,y_train.ravel())


# In[30]:


y_prediction = rfc.predict(x_test)


# In[31]:


print((accuracy_score(y_test,y_prediction)))


# # K nearest neighbours

# In[32]:


kn = KNeighborsClassifier(n_neighbors=2)
kn.fit(x_train,y_train.ravel())


# In[33]:


y_prediction = kn.predict(x_test)


# In[34]:


print((accuracy_score(y_test,y_prediction)))


# # Support Vector Machine

# In[35]:


s = SVC()
s.fit(x_train,y_train.ravel())


# In[36]:


y_prediction = s.predict(x_test)
print((accuracy_score(y_test,y_prediction)))


# # Logistic Regression

# In[37]:


lr = LogisticRegression()
lr.fit(x_train,y_train.ravel())


# In[38]:


y_prediction = lr.predict(x_test)
print((accuracy_score(y_test,y_prediction)))


# In[ ]:




