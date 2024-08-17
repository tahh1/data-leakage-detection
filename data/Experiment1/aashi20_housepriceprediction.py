#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


#Loading train data
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train_data.head()



# In[3]:


#Loading Test Data
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_data.head()


# In[4]:


train_data.describe()


# In[5]:


#separating ID as separate dataframe
train_id = train_data['Id']
test_id = test_data['Id']

train_data.drop('Id',axis = 1, inplace = True)
test_data.drop('Id',axis = 1, inplace = True)



# In[6]:


#Correlation matrix
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[7]:


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[8]:


sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)


# In[9]:


#applying log transformation
train_data['SalePrice'] = np.log(train_data['SalePrice'])

#transformed histogram and normal probability plot
sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)


# In[10]:


#saleprice correlation matrix after normalizing SalesPrice
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# Overall Qual has highest corr
# Garage Cars and Garage area are highly correlated, choose any 1
# Similarly TotalBsmtSF and 1stFlrSF are correlated, GrLivArea and TotRmsAbvGrd are correlated Use any 1
# 

# In[11]:


train_data[cols].info()
#all the top cols are numeric


# In[12]:


#scatterplot between top columns after choosing one among the strongly correlated columns
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols], size = 2.5)
plt.show();


# In[13]:


train_data.isnull().count()


# In[14]:


#missing data
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# In[15]:


#dropping cols with more than 15% missing data
train_data = train_data.drop((missing_data[missing_data['Total'] > 81]).index,1)

#missing data
train_data.head()
train_data.isnull().sum().sort_values(ascending=False)


# In[16]:


#missing data in test 
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# In[17]:


#dropping the same cols in test data also
test_data = test_data.drop((missing_data[missing_data['Total'] > 78]).index,1)

#missing data
test_data.head()
test_data.isnull().sum().sort_values(ascending=False)


# In[18]:


#GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with NA according to data description
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    train_data[col] = train_data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')


# In[19]:


#GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    train_data[col] = train_data[col].fillna('0')
    test_data[col] = test_data[col].fillna('0')


# In[20]:


#For all these categorical basement-related features, NaN means that there is no basement.
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_data[col] = train_data[col].fillna('None')
    test_data[col] = test_data[col].fillna('None')


# In[21]:


#missing values are likely zero for having no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    train_data[col] = train_data[col].fillna('0')
    test_data[col] = test_data[col].fillna('0')


# In[22]:


#NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.

train_data["MasVnrType"] = train_data["MasVnrType"].fillna("None")
train_data["MasVnrArea"] = train_data["MasVnrArea"].fillna(0)
test_data["MasVnrType"] = test_data["MasVnrType"].fillna("None")
test_data["MasVnrArea"] = test_data["MasVnrArea"].fillna(0)


# In[23]:


train_data["MSZoning"].value_counts()


# In[24]:


#RL is by far the most common value. So we can fill in missing values with 'RL'
train_data['MSZoning'] = train_data['MSZoning'].fillna(train_data['MSZoning'].mode()[0])
test_data['MSZoning'] = test_data['MSZoning'].fillna(test_data['MSZoning'].mode()[0])


# In[25]:


#Similarly, filling value for Electrical in train data
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])


# In[26]:


#Functional : data description says NA means typical
test_data["Functional"] = test_data["Functional"].fillna("Typ")


# In[27]:


train_data["MSZoning"].value_counts()


# In[28]:


#For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
#Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
train_data = train_data.drop(['Utilities'], axis=1)
test_data = test_data.drop(['Utilities'], axis=1)


# In[29]:


#KitchenQual: Only one NA value, and same as Electrical, we set 'TA'
test_data['KitchenQual'] = test_data['KitchenQual'].fillna(test_data['KitchenQual'].mode()[0])

#Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
test_data['Exterior1st'] = test_data['Exterior1st'].fillna(test_data['Exterior1st'].mode()[0])
test_data['Exterior2nd'] = test_data['Exterior2nd'].fillna(test_data['Exterior2nd'].mode()[0])
#SaleType : Fill in again with most frequent which is "WD"
test_data['SaleType'] = test_data['SaleType'].fillna(test_data['SaleType'].mode()[0])


# In[30]:


#missing data
total = test_data.isnull().sum().sort_values(ascending=False)
percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(25)


# from the pair plot above it is clear that there are outliers in GrLivArea. 
# removing those outliers
# 

# In[31]:


#exploring the outliers 
fig, ax = plt.subplots()
ax.scatter(x = train_data['GrLivArea'], y = train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[32]:


#Deleting Bottom right two outliers
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)

#Checking the scatterplot again
fig, ax = plt.subplots()
ax.scatter(train_data['GrLivArea'], train_data['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# In[33]:


#MSSubClass=The building class
train_data['MSSubClass'] = train_data['MSSubClass'].apply(str)
test_data['MSSubClass'] = test_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
train_data['OverallCond'] = train_data['OverallCond'].astype(str)
test_data['OverallCond'] = test_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
train_data['YrSold'] = train_data['YrSold'].astype(str)
test_data['MoSold'] = test_data['MoSold'].astype(str)


# In[34]:


train_data.info()


# In[35]:


from sklearn.preprocessing import LabelEncoder
cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional',  'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street',  'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train_data[c].values)) 
    train_data[c] = lbl.transform(list(train_data[c].values))
    
    


# In[36]:


train_data.head()


# In[37]:


test_data.info()


# In[38]:


cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional',  'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street',  'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(test_data[c].values)) 
    test_data[c] = lbl.transform(list(test_data[c].values))


# In[39]:


test_data.head()


# In[40]:


#train_data = pd.get_dummies(train_data)
#test_data = pd.get_dummies(test_data)


# In[41]:


test_data.head()


# In[42]:


all_data = pd.concat((train_data, test_data)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
y=train_data.SalePrice
all_data.head()


# In[43]:


categoric_feats = all_data.dtypes[all_data.dtypes == "object"].index

# process columns, apply LabelEncoder to categorical features
for c in categoric_feats:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print(('Shape all_data: {}'.format(all_data.shape)))


# In[44]:


ntrain = train_data.shape[0]
train=all_data[:ntrain]
train.head()


# In[45]:


test=all_data[ntrain:]
test.head()


# In[46]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[47]:


y_train = y


# In[48]:


column = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
train_select = train[column]
test_select = test[column]


# In[49]:


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[50]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0006, random_state=1))


# In[51]:


KRR = KernelRidge(alpha=0.5, kernel='polynomial', degree=2, coef0=2.7)


# In[52]:


GBoost = GradientBoostingRegressor(n_estimators=4000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[53]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                            learning_rate=0.05, max_depth=3, 
                            min_child_weight=1.7817, n_estimators=2900,
                            reg_alpha=0.4640, reg_lambda=0.8571,
                            subsample=0.5213, silent=1,
                            random_state =9, nthread = -1)                            


# In[54]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[55]:


score = rmsle_cv(lasso)
print(("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())))


# In[56]:


score = rmsle_cv(KRR)
print(("\nKRR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())))


# In[57]:


score = rmsle_cv(GBoost)
print(("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())))


# In[58]:


score = rmsle_cv(model_xgb)
print(("\n XGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())))


# In[59]:


score = rmsle_cv(model_lgb)
print(("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std())))


# In[60]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# In[61]:


#Final Prediction
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print((rmsle(y_train, xgb_train_pred)))


# In[62]:


#Light BGM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print((rmsle(y_train, lgb_train_pred)))


# In[63]:


ensemble = xgb_pred*0.50 + lgb_pred*0.50


# In[64]:


sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = lgb_pred
sub.to_csv('submission.csv',index=False)
print(sub)

