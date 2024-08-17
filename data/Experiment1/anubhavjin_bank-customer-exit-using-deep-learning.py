#!/usr/bin/env python
# coding: utf-8

# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 
# 
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# 
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# 
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# 
# # Any results you write to the current directory are saved as output.

# In[20]:


df=pd.read_csv('/kaggle/input/bank-customers/Churn Modeling.csv')
df.head()


# In[21]:


df.info()


# In[22]:


df.describe()


# In[23]:


X=df.iloc[:,3:13]
y=df.iloc[:,13]


# In[24]:


X.head()


# **creating dummies to remove categorical variables**

# In[25]:


states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)


# In[26]:


X=pd.concat([X,states,gender],axis=1)
X.head()


# In[28]:


X=X.drop(['Geography','Gender'],axis=1)


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# **sacling values to put in our data**

# In[31]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[33]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# **creating our ann model**

# In[35]:


classifier=Sequential()
classifier.add(Dense(activation='relu',input_dim=11,units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
classifier.summary()


# In[36]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[37]:


classifier.fit(X_train,y_train,batch_size=10,nb_epoch=50)


# In[40]:


y_pred=classifier.predict(X_test)


# In[41]:


y_pred=(y_pred>0.5)


# In[42]:


y_pred


# In[44]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)


# In[45]:


acc


# In[46]:


cm


# **had claimed an accuracy of 86%**

# now cahnging our ann futhetr

# In[52]:


classifier=Sequential()
classifier.add(Dense(activation='relu',input_dim=11,units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=9,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
classifier.summary()
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=30)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
acc=accuracy_score(y_test,y_pred)
acc


# > **IT SEEMS THAT DATA HAD BEEN OVERFITTED . WE CAN USE DROPOUT OR FLATTEN ON OUR NEURAL NET TO AVOIS OVERFITTING......our first model has max accuracy among different models tried by myself**

# In[ ]:




