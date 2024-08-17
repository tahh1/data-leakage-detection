#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


#import tensorflow as tf 
#import tensorflow.keras as tfk
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.config.experimental_connect_to_cluster(tpu)
#tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


# References: https://www.kaggle.com/masumrumi/a-detailed-regression-guide-with-house-pricing
#             https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=np.inf)
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

print('ready')


# In[4]:


df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_train.head()


# In[5]:


df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()


# In[6]:


y=df_train['SalePrice']
sns.distplot(y)


# In[7]:


#y_test=df_test['SalePrice']


# In[8]:


y=np.log1p(y)
sns.distplot(y)


# In[9]:


#y_test=np.log1p(y_test)
#sns.distplot(y_test)


# In[10]:


X=df_train.drop(['Id','SalePrice'],axis=1)
X.head()


# In[11]:


X_test=df_test.drop(['Id'],axis=1)
X_test.head()


# In[12]:


for column in X.columns:
    print((column, X[column].dtype))


# In[13]:


#missing data
total = X.isnull().sum().sort_values(ascending=False)
percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[14]:


X.PoolQC.unique()


# In[15]:


X.PoolArea.describe()


# In[16]:


X.Alley.unique()


# In[17]:


#missing data

percent_nogarage =1- X[X['GarageArea']>0]['GarageArea'].count()/X['GarageArea'].count()
percent_nogarage


# In[18]:


var='TotalBsmtSF'
percent_no =1- X[X[var]>0][var].count()/X[var].count()
percent_no


# Feature Engineearing
# 

# In[19]:


#drop variables with more than 15% missing values
X = X.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)
X.head()


# In[20]:


X_test=X_test.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)
X_test.head()


# In[21]:


X.filter(like='Yr').columns


# In[22]:


X['GarageYrBlt']=X['GarageYrBlt'].astype('str')
X['YrSold']=X['YrSold'].astype('str')
X['MSSubClass']=X['MSSubClass'].astype('str')
X['YearBuilt']=X['YearBuilt'].astype('str')
X['YearRemodAdd']=X['YearRemodAdd'].astype('str')
X['GarageYrBlt']=X['GarageYrBlt'].astype('str')


# In[23]:


X_test['GarageYrBlt']=X_test['GarageYrBlt'].astype('str')
X_test['YrSold']=X_test['YrSold'].astype('str')
X_test['MSSubClass']=X_test['MSSubClass'].astype('str')
X_test['YearBuilt']=X_test['YearBuilt'].astype('str')
X_test['YearRemodAdd']=X_test['YearRemodAdd'].astype('str')
X_test['GarageYrBlt']=X_test['GarageYrBlt'].astype('str')


# In[24]:


X[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']]=X[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']].fillna("No Garage") 
X['BsmtExposure']=X['BsmtExposure'].fillna(X['BsmtExposure'].mode()[0])
X['BsmtFinType2']=X['BsmtFinType2'].fillna(X['BsmtFinType2'].mode()[0])
X['BsmtFinType1']=X['BsmtFinType1'].fillna(X['BsmtFinType1'].mode()[0])
X['BsmtCond']=X['BsmtCond'].fillna(X['BsmtCond'].mode()[0])
X['BsmtQual']=X['BsmtQual'].fillna(X['BsmtQual'].mode()[0])
X['MasVnrType']=X['MasVnrType'].fillna(X['MasVnrType'].mode()[0])
X['MasVnrArea']=X['MasVnrArea'].fillna(X['MasVnrArea'].mean())
X['Electrical']=X['Electrical'].fillna(X['Electrical'].mode()[0])


# In[25]:


X_test[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']]=X_test[['GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual']].fillna("No Garage") 
X_test['BsmtExposure']=X_test['BsmtExposure'].fillna(X_test['BsmtExposure'].mode()[0])
X_test['BsmtFinType2']=X_test['BsmtFinType2'].fillna(X_test['BsmtFinType2'].mode()[0])
X_test['BsmtFinType1']=X_test['BsmtFinType1'].fillna(X_test['BsmtFinType1'].mode()[0])
X_test['BsmtCond']=X_test['BsmtCond'].fillna(X_test['BsmtCond'].mode()[0])
X_test['BsmtQual']=X_test['BsmtQual'].fillna(X_test['BsmtQual'].mode()[0])
X_test['MasVnrType']=X_test['MasVnrType'].fillna(X_test['MasVnrType'].mode()[0])
X_test['MasVnrArea']=X_test['MasVnrArea'].fillna(X_test['MasVnrArea'].mean())
X_test['Electrical']=X_test['Electrical'].fillna(X_test['Electrical'].mode()[0])


# In[26]:


sum(X.isnull().sum())


# In[27]:


X.head()


# In[28]:


sum(X_test.isnull().sum())


# In[29]:


X_test.head()


# In[30]:


#missing data
total_test = X_test.isnull().sum().sort_values(ascending=False)
percent_test = (X_test.isnull().sum()/X_test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)


# In[31]:


for col in X_test.columns:
    X_test[col]=X_test[col].fillna(X_test[col].mode()[0])


# In[32]:


X_test.head()


# In[33]:


sum(X_test.isnull().sum())


# No missing Values!

# In[34]:


#Log transform all float variables
for col in X.columns:
    X_test[col]=X_test[col].astype(X[col].dtypes)
    if X[col].dtypes != 'object' and X[col].max()>100 :
        X[col]=np.log1p(X[col])
        X_test[col]=np.log1p(X_test[col])
    else:
        X[col]=X[col]
        X_test[col]=X_test[col]
    


# In[35]:


X['SaleCondition'].unique()


# In[36]:


X_test['SaleCondition'].unique()


# In[37]:


columns0=X.columns


# In[38]:


columns0


# In[39]:


X=pd.get_dummies(X)
X_test=pd.get_dummies(X_test)
X, X_test = X.align(X_test, join='outer', axis=1,fill_value=0)


# In[40]:


dummycol= [col for col in X.columns if col not in columns0]


# In[41]:


dummycol


# In[42]:


sum(X.isnull().sum())


# In[43]:


X.filter(like='MSZoning')


# In[44]:


X_test.filter(like='MSZoning')


# In[45]:


X.head()


# In[46]:


X_test.head()


# In[47]:


from sklearn import metrics
from sklearn import linear_model,decomposition
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as ms
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.feature_selection import SelectFromModel
pca = decomposition.PCA()
lasso1=Lasso()
    


# In[48]:


X_train, X_test0, y_train, y_test0 = train_test_split(X, y, test_size=0.2, random_state=42)


# In[49]:


alpha_range=np.linspace(0.00001,0.05,250)
#n_components = [20, 40, 64]


# In[50]:


pipe = Pipeline(steps=[('scale',StandardScaler()),('mylasso',lasso1)])


# In[51]:


#('poly', PolynomialFeatures()),('pca',pca)


# In[52]:


#with tpu_strategy.scope():
CVlasso=GridSearchCV(pipe,dict(mylasso__alpha=alpha_range),cv=5)#pca__n_components=n_components
CVlasso.fit(X_train,y_train)


# In[53]:


bestalpha=CVlasso.best_params_['mylasso__alpha']
print(bestalpha)


# In[54]:


#pipe.set_params(mylasso__alpha=bestalpha)
#Lassoreg=pipe.fit(X_train,y_train)


# In[55]:


#Lassoreg.score(X_test0,y_test0)


# In[56]:


#y_predict0=Lassoreg.predict(X_test0)
#mse(y_test0,y_predict0)


# In[57]:


y_predict0=CVlasso.best_estimator_.predict(X_test0)
mse(y_test0,y_predict0)


# Let's try polynomial features! 

# In[58]:


poly = PolynomialFeatures(2).fit(X_train)


# In[59]:


X_train_poly= pd.DataFrame(poly.transform(X_train), columns = poly.get_feature_names(X_train.columns))


# In[60]:


X_train_poly.head()


# In[61]:


X_test0_poly= pd.DataFrame(poly.transform(X_test0), columns = poly.get_feature_names(X_train.columns))


# In[62]:


pipe_poly = Pipeline(steps=[('scale',StandardScaler()),('mylasso',lasso1)])


# In[63]:


scaler=StandardScaler()


# In[64]:


sel_ = SelectFromModel(Lasso(alpha=0.00001))
sel_.fit(scaler.fit_transform(X_train_poly), y_train)


# In[65]:


selected_feat = X_train_poly.columns[(sel_.get_support())]
print(('total features: {}'.format((X_train_poly.shape[1]))))
print(('selected features: {}'.format(len(selected_feat))))
print(('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0))))


# In[66]:


X_train_poly_select= X_train_poly[selected_feat] 


# In[67]:


alpha_range_poly=np.linspace(0.00001,0.05,250)


# In[68]:


CVlasso_poly=GridSearchCV(pipe_poly,dict(mylasso__alpha=alpha_range_poly),cv=5)#pca__n_components=n_components
CVlasso_poly.fit(X_train_poly_select,y_train)


# In[69]:


bestalpha_poly=CVlasso_poly.best_params_['mylasso__alpha']
print(bestalpha_poly)


# In[70]:


pipe_poly.set_params(mylasso__alpha=bestalpha_poly)
LassoModel_poly=pipe_poly.fit(X_train_poly_select,y_train)


# In[71]:


LassoModel_poly.score(X_test0_poly[selected_feat],y_test0)


# In[72]:


y_predict0_poly=LassoModel_poly.predict(X_test0_poly[selected_feat])
mse(y_test0,y_predict0_poly)


# Lasso without poly features seems better!

# In[73]:


sel_ = SelectFromModel(Lasso(alpha=bestalpha))
sel_.fit(scaler.fit_transform(X_train), y_train)


# In[74]:


selected_feat = X_train.columns[(sel_.get_support())]
print(('total features: {}'.format((X_train.shape[1]))))
print(('selected features: {}'.format(len(selected_feat))))
print(('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0))))


# In[75]:


selected_feat


# Let's see whether other algorithms perform better.

# In[76]:


from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# In[77]:


xgb = XGBRegressor()
xgb_p = { 'ntread':[12],
          'objective':['reg:linear'],
            'learning_rate': [0.01,0.02,0.03,0.04,0.05,0.06,0.1,0.2,0.25,0.3], 
                      'max_depth': [3,4,5,6,7,8,9,10],
                      'min_child_weight': [4],
                      'silent': [1],
                      'subsample': [0.8],
                      'colsample_bytree': [0.7],
                      'n_estimators': [500]}#'nthread':[4]


# In[78]:


xgb.fit(X_train,y_train)


# In[79]:


y_xgb=xgb.predict(X_test0)
mse(y_test0,y_xgb)


# In[80]:


CV_xgb=GridSearchCV(xgb,xgb_p,cv=5)
CV_xgb.fit(X_train[selected_feat],y_train)


# In[81]:


CV_xgb.best_params_


# In[82]:


y_xgb=CV_xgb.best_estimator_.predict(X_test0[selected_feat])
mse(y_test0,y_xgb)


# In[83]:


rf=RandomForestRegressor(random_state=42)


# In[84]:


rf_p={'n_estimators':[500],
      'max_features':['auto','sqrt'],
      'min_samples_split':[2,4,8],
      'min_samples_leaf':[3,4,5],
      'bootstrap':[True]
     }


# In[85]:


CVrf=GridSearchCV(rf,param_grid=rf_p,cv=5)
CVrf.fit(X_train[selected_feat],y_train)


# In[86]:


list(CVrf.get_params().keys())


# In[87]:


y_rf=CVrf.best_estimator_.predict(X_test0[selected_feat])
mse(y_test0,y_rf)


# In[88]:


from sklearn.linear_model import LinearRegression 


# In[89]:


pipe_linear=Pipeline(steps=[('scale',StandardScaler()),('linear',LinearRegression())])
pipe_linear.fit(X_train,y_train)
y_linear=pipe_linear.predict(X_test0)
mse(y_test0,y_linear)


# In[90]:


y_predict_lasso=CVlasso.best_estimator_.predict(X_test)


# In[91]:


y_predict_l=np.exp(y_predict_lasso) - 1


# In[92]:


y_predict_xgb=CV_xgb.best_estimator_.predict(X_test[selected_feat])


# In[93]:


y_predict_x=np.exp(y_predict_xgb) - 1


# In[94]:


y_predict=(2*y_predict_l+y_predict_x)/3


# In[95]:


sub = pd.DataFrame()
sub['Id'] = df_test['Id']
sub['SalePrice'] = y_predict
sub.to_csv('submission.csv',index=False)

