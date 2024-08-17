#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
indices_data = pd.read_csv("../input/nifty-indices-dataset/NIFTY 50.csv")


# In[2]:


indices_data.head()


# In[3]:


indices_data.columns


# In[4]:


y = indices_data.Low


# In[5]:


indices_data = indices_data.dropna(axis=0)


# In[6]:


indices_features = ['Low']


# In[7]:


X = indices_data [indices_features]


# In[8]:


X.describe()


# In[9]:


X.head()


# In[10]:


from sklearn.tree import DecisionTreeRegressor
indices_model = DecisionTreeRegressor(random_state=1)
indices_model.fit(X, y)


# In[11]:


print("The Low 5 following numbers :")
print((X.head()))
print("Numbers are")
print((indices_model.predict(X.head())))


# In[12]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = indices_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[13]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
# Fit model
indices_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = indices_model.predict(val_X)
print((mean_absolute_error(val_y, val_predictions)))


# In[14]:


from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[15]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae)))


# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print((mean_absolute_error(val_y, melb_preds)))

