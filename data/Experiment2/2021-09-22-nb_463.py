#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[26]:


data=pd.read_csv('D:/MLOPS/data.csv')


# In[27]:


data.head(10)


# In[31]:


data.drop('Index',axis='columns', inplace=True)


# In[29]:


data.rename(columns = {'Defaulted?': 'Defaulted','Bank Balance':'Bank_Balance','Annual Salary':'Annual_Salary'}, inplace = True)


# In[32]:


data


# In[33]:


data.columns


# In[34]:


X=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[35]:


y.value_counts()


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[38]:


from sklearn.ensemble import RandomForestClassifier
classifier_1=RandomForestClassifier()
classifier_1.fit(X_train,y_train)


# In[39]:


## Prediction
y_pred=classifier_1.predict(X_test)


# In[40]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


# In[41]:


score


# In[42]:


import pickle
pickle_out = open("D:/MLOPS/classifier_1.pkl","wb")
pickle.dump(classifier_1, pickle_out)
pickle_out.close()


# In[ ]:




