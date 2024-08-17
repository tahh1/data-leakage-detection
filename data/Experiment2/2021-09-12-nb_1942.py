#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


data = pd.read_csv('D:/Data Science/Assignment/16Nearal Network/gas_turbines.csv')
data.head()


# In[40]:


data['TEY'].value_counts()


# In[41]:


#every column value counts
for i in data.columns:
  print((data[i].value_counts(),'\n'))


# In[7]:


data.columns


# In[8]:


#plot AT column data
sns.boxplot(data['AT'])


# In[9]:


#plot AP column data
sns.boxplot(data['AP'])


# In[10]:


#plot AFDP column data
sns.boxplot(data['AFDP'])


# In[11]:


data['AFDP']


# In[12]:


#plot GTEP column data
sns.boxplot(data['GTEP'])
data['GTEP']=data['GTEP'].loc[data.GTEP<33]


# In[13]:


data=data.dropna(axis=0)
data.shape


# In[14]:


data.describe()


# In[15]:


#plot CO column data
sns.boxplot(data['CO'])


# In[16]:


#model build using Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_model= DecisionTreeRegressor()
x=data.drop('TEY',axis=1)
y=data['TEY']


# In[17]:


dt_model.fit(x,y)
l=dt_model.feature_importances_.round(2)>0.001
print((np.where(l==True)))


# In[18]:


new_data=x.iloc[:,[0, 5, 6, 7]]
new_data


# In[19]:


#standard scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_scaled=sc.fit_transform(new_data)
df= pd.DataFrame(x_scaled, columns=new_data.columns)
df


# In[20]:


#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=42)


# In[21]:


#keras dictonary
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adadelta,Adam
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import GridSearchCV,KFold,cross_val_score


# In[24]:


#kfold 
kfold=KFold(n_splits=10)
def create_model():
    model=Sequential([Dense(12,kernel_initializer='normal',activation='relu'),Dense(8,kernel_initializer='normal',activation='relu'),Dense(1,kernel_initializer='normal')])
    adam=Adam(lr=0.01)
    model.compile(loss='mean_squared_error',optimizer=adam)
    return model


# In[25]:


#build model
model= create_model()
model_one= model.fit(x_scaled,y,epochs=100)


# In[42]:


list(model_one.history.keys())


# In[49]:


#prediction
y_pred=model.predict(x_test)
y_pred


# In[31]:


from sklearn.metrics import r2_score


# In[51]:


#r2_score
print((r2_score(y_pred,y)))


# In[52]:


plt.scatter(y_pred, y)
plt.xlabel("predicted value")
plt.ylabel("actual value")


# In[46]:


#build model 2
model_2 = model.fit(np.array(x_train), np.array(y_train), epochs=500)


# In[47]:


#Training Acuuracy
y_train_pred = model.predict(x_train)
print(("training accuracy :", r2_score(y_train_pred, y_train)))


# In[48]:


#Testing Accuracy
y_test_pred = model.predict(x_test)
print(("training accuracy :", r2_score(y_test_pred, y_test)))


# 
