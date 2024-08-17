#!/usr/bin/env python
# coding: utf-8

# #  House Prices: Advanced Regression Techniques

# # Haythem Hsine

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb


# ## Import Data

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)
print((train.shape,test.shape))


# ## Concat train and test Data

# In[ ]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
all_data.head()


# ## Normalization

# log(price+1)

# In[ ]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[ ]:


train["SalePrice"] = np.log1p(train["SalePrice"])


# 

# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# ## Delete Missing Data

# In[ ]:


total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(22)


# In[ ]:


all_data = all_data.drop((missing_data[missing_data['Total'] > 2]).index,1)


# In[ ]:


all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
all_data.head()


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(all_data)
all_data = pd.DataFrame(data = principalComponents)


# In[ ]:


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice
print((X_train.shape,X_test.shape))


# ## CV TEST

# In[ ]:


training_set=X_train[:1200]
y_training=y[:1200]
X_CV=X_train[1200:]
y_CV=y[1200:]


# In[ ]:


regr_cv = linear_model.LinearRegression()
regr_cv.fit(training_set,y_training)
acc_log = regr_cv.score(X_CV, y_CV) * 100
acc_log


# ## forest cv

# In[ ]:


### forest 
random_forest = RandomForestRegressor(n_estimators=200)
random_forest.fit(training_set,y_training)

acc_random_forest =random_forest.score(X_CV, y_CV) * 100
acc_random_forest


# In[ ]:


random_forest.fit(X_train,y)
pred_forest=random_forest.predict(X_test)
pred_forest=np.exp(pred_forest)
pred_forest=pred_forest-1


# In[ ]:


g=GradientBoostingRegressor()
g.fit(training_set,y_training)

acc_g =g.score(X_CV, y_CV) * 100
acc_g


# In[ ]:


g.fit(X_train,y)
pred_g=random_forest.predict(X_test)
pred_g=np.exp(pred_g)
pred_g=pred_g-1


# # XGBOOST

# In[ ]:


### XGB
XGB=  XGBRegressor(seed=0,
            n_estimators=200, max_depth=10,
            learning_rate=0.05, subsample=0.3, colsample_bytree=0.75)
XGB.fit(training_set,y_training)

acc_XGB =XGB.score(X_CV, y_CV) * 100
acc_XGB


# In[ ]:


pred=XGB.predict(X_test)


# ## Modeling

# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(X_train,y)
pred = regr.predict(X_test)
pred=np.exp(pred)
pred=pred-1
pd.DataFrame({'Id': test.Id, 'SalePrice': pred}).to_csv('haythem house5.csv', index =False) 

pred

