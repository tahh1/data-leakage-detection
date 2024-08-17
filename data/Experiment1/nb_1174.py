#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# In[25]:


df=pd.read_csv(r'C:\Users\sikandar singh sekho\Desktop\College\MBA-IT\sem 3\Fundamentals of Data Science\Admission_Predict.csv')


# In[26]:


df.head()


# In[96]:


df.size


# In[55]:


df.drop(columns=['Serial No.'],inplace=True)


# In[56]:


df.head()


# In[57]:


df.describe()


# In[62]:


dataframe=df[['GRE Score','Chance of Admit ']].copy()


# In[63]:


dataframe.plot(x="GRE Score",y="Chance of Admit ",style="o")
plt.xlabel("GRE Scores")
plt.ylabel("Chances of Admit")


# In[67]:


x=dataframe.iloc[:,:-1]
y=dataframe.iloc[:,1]


# In[73]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[74]:


model=LinearRegression().fit(x_train,y_train)


# In[75]:


print((model.intercept_))


# In[76]:


print((model.coef_))


# In[77]:


y_pred=model.predict(x_test)


# In[80]:


data=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
data.head(5)


# ### Multiple Regression

# In[82]:


a=df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
b=df['Chance of Admit ']


# In[83]:


from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=0)


# In[84]:


model2 = LinearRegression().fit(a_train, b_train)


# In[87]:


coeff_df = pd.DataFrame(model2.coef_, a.columns, columns=['Coefficient'])
coeff_df


# In[89]:


intercept_df=pd.DataFrame(model2.intercept_,a.columns,columns=['Intercept'])
intercept_df


# In[90]:


b_pred = model2.predict(a_test)


# In[92]:


data2 = pd.DataFrame({'Actual': b_test, 'Predicted': b_pred})
data2.head()


# In[ ]:




