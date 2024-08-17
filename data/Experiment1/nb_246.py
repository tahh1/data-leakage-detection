#!/usr/bin/env python
# coding: utf-8

# # Feature Selection

# ### Import Libraries

# In[111]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


#!pip install skopt


pd.set_option('display.max_columns',None)


# ### Import datasets

# In[112]:


data=pd.read_csv('train_test.csv')
train = data.iloc[:1460,:]
target = train['SalePrice']
train.drop('SalePrice',axis=1,inplace=True)
test = data.iloc[1460:,:]
test.drop('SalePrice',axis=1,inplace=True)


# In[113]:


train.shape,test.shape


# In[114]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesRegressor


# In[115]:


df=train.copy()
df=pd.concat([df,target],axis=1)


# In[116]:


cor=df.corr()['SalePrice']


# In[121]:


Final_features = cor[abs(cor)>0.1].index
Final_features = Final_features.drop('SalePrice')


# In[122]:


train = train[Final_features]
test = test[Final_features]


# In[123]:


Final_features


# In[125]:


from sklearn.model_selection import RandomizedSearchCV


######################################################
####### Hyper parameter Optimization

n_estimators=[100,500,900,1100,1500]
max_depth=[2,3,5,10,15]
#booster=['gbtree','bglinear']
learning_rate = [0.05,0.1,0.15,0.2]
min_child_weight=[1,2,3,4]

######################################################
######  Define the grid of hyperparameters to search
hyperparameter_grid={
    'n_estimators':n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    #'booster':booster,
    #'base_score':base_score
}


random_cv = RandomizedSearchCV(estimator=reg,param_distributions=hyperparameter_grid,
                              cv=5,n_iter=50,
                                scoring='neg_mean_absolute_error',n_jobs=4,
                              verbose=5,
                              return_train_score=True,
                              random_state=42)


# In[126]:


random_cv.fit(train,target)


# In[108]:


random_cv.best_estimator_


# In[124]:


import xgboost
from sklearn.linear_model import LinearRegression
reg = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=10,
             min_child_weight=3, missing=np.nan, monotone_constraints='()',
             n_estimators=900, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
reg.fit(train,target)

y_pred = np.exp(reg.predict(test))


sample_sub = pd.read_csv('sample_submission.csv')
ID2=sample_sub.Id
submission_table=pd.DataFrame({'Id':ID2,'SalePrice':y_pred})
submission_table.to_csv('Submission_feature',index=False)
print('Done!')


# In[ ]:




