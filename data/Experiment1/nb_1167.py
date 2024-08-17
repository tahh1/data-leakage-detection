#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_excel(r'Data_Train.xlsx')


# In[3]:


pd.set_option('display.max.columns',None)


# In[4]:


train_data.head(3)


# In[5]:


train_data.shape


# In[6]:


print((train_data.index))
print((train_data.columns))


# In[7]:


train_data.info()


# In[8]:


train_data.describe()


# In[9]:


train_data.isnull().sum()


# In[10]:


train_data['Airline'].value_counts()


# In[11]:


train_data['Duration'].value_counts()


# In[12]:


train_data.dropna(inplace=True)


# In[13]:


train_data.isnull().sum()


# In[14]:


train_data.shape


# # EDA Of Data

# In[15]:


train_data['journy_day'] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data["journy_month"] = pd.to_datetime(train_data["Date_of_Journey"],format="%d/%m/%Y").dt.month
train_data.drop("Date_of_Journey", axis=1, inplace=True)


# In[16]:


train_data.head(3)


# In[17]:


train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data["Dep_minute"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
train_data.drop("Dep_Time", axis=1, inplace=True)


# In[18]:


train_data.head()


# In[19]:


train_data["Arrival_hour"] = pd.to_datetime(train_data["Arrival_Time"]).dt.hour
train_data["Arrival_minute"] = pd.to_datetime(train_data["Arrival_Time"]).dt.minute
train_data.drop("Arrival_Time", axis=1, inplace=True)


# In[20]:


train_data.head(3)


# In[21]:


duration = list(train_data["Duration"])


# In[22]:


for i in range(len(duration)):
    if len(duration[i].split()) !=2 :
        if 'h' in duration[i]:
            duration[i] = duration[i].strip()+" 0m"
        else:
            duration[i] = "0h " + duration[i]
            
duration_hour =[]
duration_minute = []

for i in range(len(duration)):
    duration_hour.append(int(duration[i].split(sep="h")[0]))
    duration_minute.append(int(duration[i].split(sep="m")[0].split()[-1]))
    


# In[23]:


train_data['duration_hour'] = duration_hour
train_data['duration_minute'] = duration_minute


# In[24]:


train_data.head()


# In[25]:


train_data.drop("Duration", axis=1, inplace=True)


# In[26]:


train_data.head()


# # Categorical Data handling

# In[27]:


train_data['Airline'].value_counts()


# In[28]:


Airline = train_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


# In[29]:


Source = train_data[['Source']]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()


# In[30]:


train_data.head()


# In[31]:


Destination = train_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
Destination.head()


# In[32]:


train_data.drop(["Route","Additional_Info"], axis=1, inplace=True)


# In[33]:


train_data.head(3)


# In[34]:


train_data['Total_Stops'].value_counts()


# In[35]:


train_data.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}, inplace=True)


# In[36]:


train_data = pd.concat([train_data,Destination,Source,Airline],axis=1)


# In[37]:


train_data.head(4)


# In[38]:


train_data.drop(["Airline","Source",'Destination'], axis=1, inplace=True)


# In[39]:


train_data.head()


# In[40]:


train_data.shape


# # test data

# In[41]:


test_data = pd.read_excel(r'Test_set.xlsx')


# In[42]:


test_data['journy_day'] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["journy_month"] = pd.to_datetime(test_data["Date_of_Journey"],format="%d/%m/%Y").dt.month
test_data.drop("Date_of_Journey", axis=1, inplace=True)


# In[43]:


test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_minute"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop("Dep_Time", axis=1, inplace=True)


# In[44]:


test_data["Arrival_hour"] = pd.to_datetime(test_data["Arrival_Time"]).dt.hour
test_data["Arrival_minute"] = pd.to_datetime(test_data["Arrival_Time"]).dt.minute
test_data.drop("Arrival_Time", axis=1, inplace=True)


# In[45]:


duration = list(test_data["Duration"])
for i in range(len(duration)):
    if len(duration[i].split()) !=2 :
        if 'h' in duration[i]:
            duration[i] = duration[i].strip()+" 0m"
        else:
            duration[i] = "0h " + duration[i]
            
duration_hour =[]
duration_minute = []

for i in range(len(duration)):
    duration_hour.append(int(duration[i].split(sep="h")[0]))
    duration_minute.append(int(duration[i].split(sep="m")[0].split()[-1]))


# In[46]:


test_data.drop("Duration", axis=1, inplace=True)


# In[47]:


Airline = test_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()


# In[48]:


Source = test_data[['Source']]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()


# In[49]:


Destination = test_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
Destination.head()


# In[50]:


test_data.drop(["Route","Additional_Info"], axis=1, inplace=True)


# In[51]:


test_data.replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4}, inplace=True)


# In[52]:


test_data = pd.concat([test_data,Destination,Source,Airline],axis=1)


# In[53]:


test_data.head()


# In[54]:


test_data.drop(["Airline","Source",'Destination'], axis=1, inplace=True)


# In[55]:


test_data.head()


#  # Feauters Selection

# In[56]:


train_data.head(3)


# In[57]:


y = train_data['Price']
x = train_data.drop('Price',axis=1)


# In[58]:


y


# In[59]:


x


# In[61]:


# important feauter selection

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(x,y)


# In[63]:


print((selection.feature_importances_))


# In[64]:


# train test spliting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[65]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)


# In[66]:


model.score(x_train,y_train)


# In[67]:


model.score(x_test,y_test)


# In[68]:


y_pred = model.predict(x_test)


# In[69]:


y_pred


# In[70]:


y_test


# In[72]:


from sklearn import metrics

print((metrics.mean_absolute_error(y_test,y_pred)))


# In[73]:


print((metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:




