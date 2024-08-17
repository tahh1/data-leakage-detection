#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import pickle


# In[2]:


df=pd.read_csv("heart.csv")


# In[3]:


df.head()


# In[40]:


df["ca"].value_counts()


# In[41]:


df.columns


# In[ ]:





# In[4]:


df.describe()


# In[5]:


from sklearn.feature_selection import VarianceThreshold


# In[6]:


variance_thres=VarianceThreshold(threshold=0)
variance_thres.fit(df)


# In[7]:


variance_thres.get_support()


# In[8]:


constant_columns=[column for column in df.columns if column not in df.columns[variance_thres.get_support()]]


# In[9]:


constant_columns


# In[10]:


df.drop(constant_columns, axis=1, inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


import seaborn as sns
sns.countplot(df["age"])


# In[13]:


df["target"].value_counts()

X=df.drop(["target"], axis=1)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X, df["target"], test_size=.25, random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


lr=LogisticRegression()


# In[18]:


lr.fit(X_train,y_train)


# In[19]:


y_pred=lr.predict(X_test)


# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[21]:


cf=confusion_matrix(y_test,y_pred)


# In[22]:


print((accuracy_score(y_test,y_pred)))


# In[51]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[43]:


rf=RandomForestClassifier()


# In[44]:


rf.fit(X_train,y_train)


# In[45]:


y_pred_rf=rf.predict(X_test)


# In[46]:


cf1=confusion_matrix(y_test,y_pred_rf)


# In[47]:


print(cf1)


# In[56]:


error_rate=[]

# We will run from k1 to k40. will take time to run

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))


# In[58]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10,6))
plt.plot(list(range(1,40)),error_rate, color="blue",linestyle="dashed",marker='o',markerfacecolor='red',markersize=10)
plt.title("Error rate vs k value")
plt.xlabel("K")
plt.ylabel("Error rate")


# In[59]:


knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
pred_i=knn.predict(X_test)


# In[60]:


cf2=confusion_matrix(y_test,pred_i)


# In[61]:


print(cf2)


# In[25]:


# save the model to disk
filename = 'heart_model.pkl'
pickle.dump(lr, open(filename, 'wb'))


# In[ ]:




