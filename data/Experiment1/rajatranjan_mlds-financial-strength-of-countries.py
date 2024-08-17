#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print((os.listdir("../input/mlds_final_data/MLDS_final_data")))


# In[2]:


train=pd.read_csv('../input/mlds_final_data/MLDS_final_data/final_train_data.csv')
test1=pd.read_csv('../input/mlds_final_data/MLDS_final_data/final_test_data.csv')
train.head()


# In[3]:


test1.head()


# In[4]:


# train[train['Country Name']=='India']


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.distplot(train['Balance'],bins=100)
# np.cbrt(train['Balance'])


# In[6]:


# (1-train['Trade']/100)*train['Exports']
train.describe(include='all').T


# In[7]:


test=test1.copy()
test.describe(include='all').T


# In[8]:


train['gdp_growth']=train['GDP']-train['Inflation']
train['exports_tradeoff']=(1-train['Trade']/100)*train['Exports']
# train['gdp_tradeoff']=train['GDP']/train['Trade']

test['gdp_growth']=test['GDP']-test['Inflation']
test['exports_tradeoff']=(1-test['Trade']/100)*test['Exports']
# test['gdp_tradeoff']=test['GDP']/test['Trade']

train['Year']=train['Year'].astype(np.object)
test['Year']=test['Year'].astype(np.object)


# In[9]:


train.head()


# In[10]:


# train=train[(train['Inflation']<=4107.3) & (train['GDP']<=27) & (train['Trade']<=350) & (train['gdp_growth']<=31)]


# In[11]:


train.shape
n_cols=['Inflation', 'GDP', 'Exports', 'Trade', 'gdp_growth','exports_tradeoff']


# In[12]:


for i in n_cols:
    train[i].fillna(train[i].mean(),inplace=True)
    test[i].fillna(test[i].mean(),inplace=True)


# In[13]:


train.head()


# In[14]:


from sklearn.preprocessing import RobustScaler,StandardScaler,Normalizer
r=RobustScaler()
train[n_cols]=r.fit_transform(train[n_cols])
test[n_cols]=r.transform(test[n_cols])


# In[15]:


# from sklearn.preprocessing import RobustScaler,StandardScaler,Normalizer
# for i in ['Inflation', 'GDP', 'Exports', 'Trade', 'gdp_growth','exports_tradeoff']:
#     r=RobustScaler()
#     train[i]=r.fit_transform(train[i])
#     test[i]=r.transform(test[i])
train[n_cols].isnull().sum()


# In[16]:


test.info()


# In[17]:


from catboost import CatBoostClassifier,Pool, cv,CatBoostRegressor
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.stats import mode


# In[18]:


X_train,X_val,y_train,y_val = train_test_split(train.drop(['Unnamed: 0','Balance'],axis=1),train['Balance'],
                                                 test_size=0.25,random_state = 1994)


# In[19]:


categorical_features_indices = np.where(X_train.dtypes =='object')[0]
categorical_features_indices


# In[20]:


X,y=train.drop(['Unnamed: 0','Balance'],axis=1),train['Balance']
Xtest=test.drop(['Unnamed: 0'],axis=1)
# X.head()


# In[21]:


import math
def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return 1/(1+math.exp(np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())))

def runCatBoost(x_train, y_train,x_test, y_test,test,depth):
    model=CatBoostRegressor(n_estimators=1000,
                            learning_rate=0.1,
                            loss_function='RMSE',
                            eval_metric='RMSE',
                            random_seed=1994,
#                             od_type='Iter',
#                             od_wait=50
                           )
                           
    model.fit(x_train, y_train,cat_features=categorical_features_indices, eval_set=(x_test, y_test), use_best_model=True, verbose=150)
    y_pred_train=model.predict(x_test)
    rmsle_result = rmsle(y_pred_train,y_test)
    y_pred_test=model.predict(test)
    return y_pred_train,rmsle_result,y_pred_test
#     return y_pred_train,y_pred_test


# In[22]:


from sklearn import model_selection
pred_full_test_cat_feen = 0
mse_cat_list_feen=[]
kf = model_selection.KFold(n_splits=2, shuffle=True, random_state=30)
for dev_index, val_index in kf.split(X):
    dev_X, val_X = X.loc[dev_index], X.loc[val_index]
    dev_y, val_y = y.loc[dev_index], y.loc[val_index]
    y_pred_feen,rmsle_feen,y_pred_test_feen=runCatBoost(dev_X, dev_y, val_X, val_y,Xtest,depth=4)
    print(('fold score :',rmsle_feen))
    mse_cat_list_feen.append(rmsle_feen)
    pred_full_test_cat_feen = pred_full_test_cat_feen + y_pred_test_feen
mse_cat_feen_mean=np.mean(mse_cat_list_feen)
print(("Mean cv score : ", np.mean(mse_cat_feen_mean)))
y_pred_test_feen=pred_full_test_cat_feen/2


# In[23]:


sns.distplot(y_pred_test_feen)


# In[24]:


test1['Balance']=y_pred_test_feen
test1.drop('Unnamed: 0',axis=1,inplace=True)


# In[25]:


# y_pred_test_feen
# s=pd.DataFrame({'Balance':y_pred_test_feen})
test1.to_excel('2foldcb8.xlsx',index=False)
test1.head()


# In[26]:


# cat_model = CatBoostRegressor(n_estimators=1000, # use large n_estimators deliberately to make use of the early stopping
# #                          reg_lambda=1.0,
# #                          l2_leaf_reg=4.0,
#                          eval_metric='RMSE',
#                          random_seed=1994,
# #                          learning_rate = 0.05,
# #                          depth = 8,
                               
# #                                boosting_type = 'Ordered',
# #                          subsample = 0.8
#                          #rsm = 0.7,
#                          #silent=True,
#                          #max_ctr_complexity = 5,  # no of categorical cols combined
# #                          boosting_type = 'Ordered',
# #                          od_type = 'IncToDec',  #overfitting params
# #                          od_wait = 20)
#                          #bagging_temperature = 1.0)
#                               )
# # lr=0.05, no od type of vars -- highest
    
# cat_model.fit(X_train.values,y_train.values,cat_features=categorical_features_indices,eval_set=(X_val, y_val),
#         plot=False,early_stopping_rounds=100,use_best_model=True) 


# In[27]:


# sorted(zip(cat_model.feature_importances_,X_train),reverse=True)


# In[28]:


# X,y=train1.drop(['Unnamed: 0','Balance'],axis=1),train1['Balance']
# Xtest=test.drop(['Unnamed: 0'],axis=1)


# In[29]:


# cat_model.fit(X,y,cat_features=categorical_features_indices,eval_set=(X, y),
#         plot=False,early_stopping_rounds=100,use_best_model=True)
# y_pred=cat_model.predict(Xtest)


# In[30]:


# test.shape


# In[31]:


# s=test['Unnamed: 0']
# s['Balance']=y_pred
# test
# np.power(y_pred,3)


# In[32]:


# s=pd.DataFrame({'Balance':y_pred})
# s.head()
# # s.to_csv('catboost1.csv',index=False) --main 62 max depth-8 lr-0.1
# # s.to_excel('catboost8_4folds.xlsx',index=False) #--nope


# In[33]:


# s.to_excel('s7.xlsx',index=False)


# In[34]:


# sns.distplot(np.power(y_pred,3))


# In[35]:


# sns.distplot(train1['Balance'])


# In[36]:




