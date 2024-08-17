#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# In[126]:


df=pd.read_csv("bank-full.csv",delimiter=";")
df.head()


# In[127]:


df["job"].value_counts()


# In[128]:


df["job"]=df["job"].map({"admin.":1,"unknown":2,"unemployed":3,"management":4,"housemaid":5,"entrepreneur":6,"student":7,
                                       "blue-collar":8,"self-employed":9,"retired":10,"technician":11,"services":12})


# In[129]:


df["marital"].value_counts()


# In[130]:


df["marital"]=df["marital"].map({"married":1,"divorced":2,"single":3})


# In[131]:


df["education"].value_counts()


# In[132]:


df["education"]=df["education"].map({"unknown":1,"secondary":2,"primary":3,"tertiary":4})


# In[133]:


df["default"]=df["default"].map({"yes":1,"no":0})


# In[134]:


df["housing"]=df["housing"].map({"yes":1,"no":0})


# In[135]:


df["loan"]=df["loan"].map({"yes":1,"no":0})


# In[136]:


df["contact"].value_counts()


# In[137]:


df["contact"]=df["contact"].map({"cellular":1,"unknown":2,"telephone":3})


# In[138]:


df["month"]=df["month"].map({"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12})


# In[139]:


df["poutcome"].value_counts()


# In[140]:


df["poutcome"]=df["poutcome"].map({"unknown":1,"failure":2,"other":3,"success":4})


# In[141]:


df["y"]=df["y"].map({"yes":1,"no":0})


# In[142]:


df.info()


# In[143]:


df.head()


# In[144]:


df.shape


# In[145]:


x=df.iloc[:,:16]
y=df.iloc[:,16:]


# In[146]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)


# In[147]:


model = LogisticRegression()


# In[148]:


model.fit(x_train,y_train)


# In[149]:


y_test_pred=model.predict(x_test)


# In[117]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_test_pred)


# In[118]:


from sklearn.metrics import accuracy_score as ac
ac(y_test,y_test_pred)


# In[119]:


model.coef_


# In[120]:


model.intercept_


# In[30]:


from sklearn.metrics import classification_report
print((classification_report(y_test,y_test_pred)))


# In[31]:


y_score=model.predict_proba(x_test)[:,1]
from sklearn.metrics import roc_curve,roc_auc_score

fpr,tpr,thresholds=roc_curve(y_test,y_score)
plt.plot(fpr,tpr,color="red")
plt.plot([0,1],[0,1],"k--")
roc_auc_score(y_test,y_score)

