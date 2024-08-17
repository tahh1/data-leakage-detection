#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


test.shape


# In[7]:


train.shape


# In[8]:


#Data wrangling of train data


# In[9]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[10]:


train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['BsmtCond'] = train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])
train['BsmtQual'] = train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
train['FireplaceQu'] = train['FireplaceQu'].fillna(train['FireplaceQu'].mode()[0])
train['GarageType'] = train['GarageType'].fillna(train['GarageType'].mode()[0])
train['GarageFinish'] = train['GarageFinish'].fillna(train['GarageFinish'].mode()[0])
train['GarageQual'] = train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageCond'] = train['GarageCond'].fillna(train['GarageCond'].mode()[0])
train['BsmtExposure'] = train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0])


# In[11]:


train.drop(['PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt'], axis = 1, inplace = True)
train['BsmtFinType2'] = train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0])
sns.heatmap(train.isnull(), yticklabels = False, cbar = False)


# In[12]:


train.shape


# In[13]:


train.drop(['Alley', 'Id'], axis = 1, inplace = True)


# In[14]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False)


# In[15]:


train.dropna(inplace = True)


# In[16]:


train.shape


# In[17]:


train.columns


# In[18]:


columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 
          'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 
          'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofMatl', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
          'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
          'GarageQual', 'GarageCond', 'PavedDrive']


# In[19]:


len(columns)


# In[20]:


def category_onehot_multcols(multcolumns):
    train_final = final_data
    i = 0
    for fields in multcolumns:
        
        print(fields)
        train1 = pd.get_dummies(final_data[fields], drop_first = True)
        final_data.drop([fields], axis = 1, inplace = True)
        if i == 0:
            train_final = train1.copy()
        else:
            train_final = pd.concat([train_final, train1], axis = 1)
        i = i+1
        
    train_final = pd.concat([final_data, train_final], axis = 1)
    return train_final


# In[21]:


main_train = train.copy()


# In[22]:


#Data wrangling of test data


# In[23]:


test.isnull().sum()


# 

# In[24]:


test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())


# In[25]:


test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])


# In[26]:


test.columns


# In[27]:


test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])
test['BsmtFinType1'] = test['BsmtFinType1'].fillna(test['BsmtFinType1'].mode()[0])
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mean())
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())


# In[28]:


test.shape


# In[29]:


test['BsmtCond'] = test['BsmtCond'].fillna(test['BsmtCond'].mode()[0])
test['BsmtQual'] = test['BsmtQual'].fillna(test['BsmtQual'].mode()[0])
test['FireplaceQu'] = test['FireplaceQu'].fillna(test['FireplaceQu'].mode()[0])
test['GarageType'] = test['GarageType'].fillna(test['GarageType'].mode()[0])


# In[30]:


test.drop(['GarageYrBlt'], axis = 1, inplace = True)


# In[31]:


test['MasVnrType'] = test['MasVnrType'].fillna(test['MasVnrType'].mode()[0])
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['BsmtExposure'] = test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0])
test['BsmtFinType2'] = test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0])


# In[32]:


test.drop(['Id', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)


# In[33]:


test.shape


# In[34]:


sns.heatmap(test.isnull(), yticklabels = False, cbar = False)


# In[35]:


test.drop(['Alley'], axis = 1, inplace = True)


# In[36]:


test.shape


# In[37]:


final_data = pd.concat([train, test], axis = 0)


# In[38]:


final_data.shape


# In[39]:


final_data = category_onehot_multcols(columns)


# In[40]:


final_data = final_data.loc[:,~final_data.columns.duplicated()]


# In[41]:


final_data.shape


# In[42]:


final_data.isnull().sum()


# In[43]:


df_train = final_data.iloc[:1422,:]
df_test = final_data.iloc[1422:,:]


# In[44]:


df_test.drop(['SalePrice'], axis = 1, inplace = True)


# In[45]:


df_test.shape


# In[46]:


df_train.dropna(inplace = True)


# In[47]:


x_train = df_train.drop(['SalePrice'], axis = 1)
y_train = df_train['SalePrice']


# In[48]:


from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 324)


# In[50]:


price_classifier = DecisionTreeClassifier(max_leaf_nodes = 10, random_state = 0)
price_classifier.fit(x_train, y_train)


# In[51]:


y_predicted = price_classifier.predict(x_test)


# In[52]:


accuracy_score(y_test, y_predicted)*100


# In[53]:


y_predicted


# In[ ]:





# In[54]:


import xgboost as xg

classifier = xg.XGBRegressor()
classifier.fit(x_train, y_train)
# In[55]:


classifier = xg.XGBRegressor()
classifier.fit(x_train, y_train)


# In[56]:


y_pred = classifier.predict(df_test)


# In[57]:


y_pred


# In[58]:


pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets = pd.concat([sub_df['Id'], pred], axis = 1)
datasets.columns = ['Id', 'SalePrice']
datasets.to_csv('sample_submission_copy.csv', index = False)


# In[ ]:




