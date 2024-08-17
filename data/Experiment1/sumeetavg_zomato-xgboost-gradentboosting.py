#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("../input/zomato.csv", encoding = 'ISO-8859-1')


# In[3]:


df.head(10)


# In[4]:


to_drop = ["Locality", "Address", "Locality Verbose", "Longitude", "Latitude" , "Switch to order menu"]
df.drop(to_drop, inplace = True, axis =1 )


# In[5]:


df['Restaurant ID'].is_unique


# In[6]:


df["Country Code1"]=df["Country Code"].apply(str)
df['Country Code']=df['Country Code'].replace({189:'Canada',216:'Tunisia',215:'Philadelphia',214:'Dallas',1:'India',30:'Greece',148:'Equador'})
df['Country Code']=df['Country Code'].replace([208,14,94,191,162,184,166,37],'Others')
df=df.rename(columns={"Country Code":"Country Name"})


# In[7]:


df[df["Average Cost for two"]>450000]


# In[8]:


df=df[df["Restaurant ID"] != 7402935]
df=df[df["Restaurant ID"] != 7410290]
df=df[df["Restaurant ID"] != 7420899]


# In[9]:


df['Has Table booking'] = pd.get_dummies(df["Has Table booking"],drop_first=True)
df['Has Online delivery'] = pd.get_dummies(df["Has Online delivery"],drop_first=True)
df['Is delivering now'] = pd.get_dummies(df["Is delivering now"],drop_first=True)


# In[10]:


df['Currency']=df['Currency'].replace({'Dollar($)':'Dollar','Pounds(��)':'Pounds','Brazilian Real(R$)':'Brazilian Real','NewZealand($)':'NewZealand Dollar'})


# In[11]:


cus=df["Cuisines"].value_counts()
cuisines = {}
cnt=0
for i in cus.index:
    for j in i.split(", "):
        if j not in cuisines:
            cuisines[j]=cus[cnt]
        else:
            cuisines[j] += cus[cnt]
    cnt += 1
    
cuisines = pd.Series(cuisines).sort_values(ascending=False)


# In[12]:


India=df[df.Currency == 'Indian Rupees(Rs.)']


# In[13]:


q3_v=India["Votes"].quantile(0.75)
q1_v=India["Votes"].quantile(0.25)
iqr_v=q3_v-q1_v
lowervotes=q1_v-(iqr_v*1.5)
uppervotes=q3_v+(iqr_v*1.5)
uppervotes


# In[14]:


India=India[India["Votes"]<244]


# In[15]:


q3_avg=India["Average Cost for two"].quantile(0.75)
q1_avg=India["Average Cost for two"].quantile(0.25)
iqr_avg=q3_avg-q1_avg
loweravg=q1_avg-(iqr_avg*1.5)
upperavg=q3_avg+(iqr_avg*1.5)
upperavg


# In[16]:


India=India[India["Average Cost for two"]<1050]


# In[17]:


X=India.drop(["Restaurant ID","Restaurant Name","Rating text","Country Name","City","Rating color",
           "Cuisines","Currency","Country Code1","Aggregate rating"],axis=1)
y=India["Aggregate rating"]


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[19]:


X_train, X_test, y_train, y_test  = train_test_split(X, y , test_size = 0.2, random_state = 42)


# In[20]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[21]:


y_predict =  model.predict(X_test)


# In[22]:


from sklearn.metrics import r2_score


# In[23]:


r2_score(y_test, y_predict)


# In[24]:


from sklearn.tree import DecisionTreeRegressor



# In[25]:


modeldt= DecisionTreeRegressor(max_depth=6)
modeldt.fit(X_train,y_train)


# In[26]:


y_predictdt=modeldt.predict(X_test)
r2_score(y_test,y_predictdt)


# In[27]:


from sklearn.ensemble import GradientBoostingRegressor


# In[28]:


est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
      max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)


# In[29]:


y_predictdt=est.predict(X_test)
r2_score(y_test,y_predictdt)


# In[30]:


import xgboost as xgb


# In[31]:


xgb_clf = xgb.XGBRegressor(max_depth=3, n_estimators=5000, learning_rate=0.2,
                            n_jobs=-1)


# In[32]:


xgb_clf.fit(X_train, y_train)



# In[33]:


y_pred = xgb_clf.predict(X_test)


# In[34]:


r2_score(y_test, y_pred) 


# In[35]:




