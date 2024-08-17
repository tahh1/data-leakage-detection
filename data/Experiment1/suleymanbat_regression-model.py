#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv('../input/advertising/Advertising.csv', index_col = 0)


# In[6]:


df.head()


# In[7]:


df.describe().T


# In[8]:


df.info()


# In[9]:


import seaborn as sns
sns.jointplot(x = 'TV',y = 'sales', data = df, kind ='reg');


# In[10]:


df.head()


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


X = df[['TV']]
X.head()


# In[13]:


y = df[['sales']]
y.head()


# # MODEL**

# In[14]:


reg_model = LinearRegression()
reg_model.fit(X,y)


# In[15]:


reg_model


# In[16]:


reg_model.intercept_


# In[17]:


reg_model.coef_


# In[18]:


reg_model.score(X,y)


# # *GUESS*

# **How much money we would earn, if we were to spend 40 units on TV?**

# y_hat = 7.03 + 0.04*xi

# In[19]:


7.03 + 0.04*40


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
g = sns.regplot(df['TV'], df['sales'],ci = None, scatter_kws={'color':'r','s':9})
g.set_title('Model Denklemi: Sales = 7.03 + TV*0.05')
g.set_ylabel('Satis Sayisi')
g.set_xlabel('TV Harcamalari')
plt.xlim(-10,310)
plt.ylim(bottom=0);


# In[21]:


reg_model.intercept_ + reg_model.coef_*165


# In[22]:


reg_model.predict([[165]])


# In[23]:


new_data = [[5],[15],[30]]
reg_model.predict(new_data)


# # > MULTIPLE LINEAR REGRESSION

# In[24]:


X = df.drop('sales', axis=1)
y = df[['sales']]


# # **MODEL**

# In[25]:


reg_model = LinearRegression()


# In[26]:


reg_model.fit(X,y)


# In[27]:


reg_model.intercept_


# In[28]:


reg_model.coef_


# # **GUESS**

# # Sales = 2.94 + TV 0.04 + radio 0.19 - newspaper*0.001

# # 30 unit TV, 10 unit radio, 40 unit newspaper

# In[29]:


2.94 + 30*0.04 + 10*0.19 - 40*0.001


# In[30]:


new_dat = [[300],[120],[400]]
new_dat = pd.DataFrame(new_dat).T


# In[31]:


reg_model.predict(new_dat)


# In[32]:


from sklearn.metrics import mean_squared_error
y.head()


# # > EVALUATING OF THE SUCCESS OF GUESS MENTIONED

# In[33]:


reg_model


# In[34]:


reg_model.predict(X)


# In[35]:


y_pred = reg_model.predict(X)


# In[36]:


mse = mean_squared_error(y, y_pred)
mse


# In[37]:


import numpy as np
rmse = np.sqrt(mse)
rmse


# In[38]:


df.describe().T


# # > MODEL VALIDATION...HOLD OUT METHOD

# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)


# In[40]:


X_train.head()


# In[41]:


y_train.head()


# In[42]:


X_test.head()


# In[43]:


y_test.head()


# In[44]:


X_train.shape


# In[45]:


X_test.shape


# In[46]:


reg_model = LinearRegression()
reg_model.fit(X_train, y_train)


# In[47]:


y_pred = reg_model.predict(X_train)


# # *TRAIN ERROR*

# In[48]:


np.sqrt(mean_squared_error(y_train, y_pred))


# In[49]:


y_pred = reg_model.predict(X_test)


# # *TEST ERROR*

# In[50]:


np.sqrt(mean_squared_error(y_test, y_pred))


# # **K FOLD CROSS VALIDATION**

# In[51]:


from sklearn.model_selection import cross_val_score


# In[52]:


reg_model = LinearRegression()


# In[54]:


#FIRST METHOD:
-cross_val_score(reg_model,X,y,cv=10, scoring = 'neg_mean_squared_error')


# In[55]:


np.mean(-cross_val_score(reg_model,X,y,cv=10, scoring = 'neg_mean_squared_error'))


# In[56]:


np.std(-cross_val_score(reg_model, X,y,cv=10,scoring = 'neg_mean_squared_error'))


# In[57]:


np.sqrt(np.mean(-cross_val_score(reg_model,X,y,cv=10, scoring = 'neg_mean_squared_error')))


# # > SECOND METHOD

# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20, random_state = 1)


# In[61]:


reg_model = LinearRegression()
reg_model.fit(X_train,y_train)


# In[60]:


np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv=10, scoring = 'neg_mean_squared_error')))


# In[62]:


#Test
y_pred = reg_model.predict(X_test)


# In[63]:


#Test_Error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[ ]:




