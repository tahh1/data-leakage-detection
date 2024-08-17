#!/usr/bin/env python
# coding: utf-8

# # **Looking at house sale information to predict house prices**
# 
# We look at the housing prices dataset and see if we can predict house prices

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # better looking graphs

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

get_ipython().run_line_magic('matplotlib', 'inline')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


# Reading the data
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()


# In[3]:


df_train.columns


# # **Trying to fill in the missing data**

# In[4]:


plt.figure(figsize=(20,8))
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[5]:


df_train['Alley'].fillna(value='No alley access',inplace=True)
df_train['BsmtQual'].fillna(value='No Basement',inplace=True)
df_train['BsmtCond'].fillna(value='No Basement',inplace=True)
df_train['BsmtExposure'].fillna(value='No Basement',inplace=True)
df_train['BsmtFinType1'].fillna(value='No Basement',inplace=True)
df_train['BsmtFinType2'].fillna(value='No Basement',inplace=True)
df_train['FireplaceQu'].fillna(value='No Fireplace',inplace=True)
df_train['GarageType'].fillna(value='No Garage',inplace=True)
df_train['GarageYrBlt'].fillna(value=0,inplace=True)
df_train['GarageFinish'].fillna(value='No Garage',inplace=True)
df_train['GarageQual'].fillna(value='No Garage',inplace=True)
df_train['GarageCond'].fillna(value='No Garage',inplace=True)
df_train['MasVnrType'].fillna(value='None',inplace=True)
df_train['MasVnrArea'].fillna(value=0.0,inplace=True)
df_train['PoolQC'].fillna(value='No Pool',inplace=True)
df_train['Fence'].fillna(value='No Fence',inplace=True)
df_train['MiscFeature'].fillna(value='None',inplace=True)

plt.figure(figsize=(20,8))
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[6]:


plt.figure(figsize=(20,8))
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[7]:


df_test['Alley'].fillna(value='No alley access',inplace=True)
df_test['BsmtQual'].fillna(value='No Basement',inplace=True)
df_test['BsmtCond'].fillna(value='No Basement',inplace=True)
df_test['BsmtExposure'].fillna(value='No Basement',inplace=True)
df_test['BsmtFinType1'].fillna(value='No Basement',inplace=True)
df_test['BsmtFinType2'].fillna(value='No Basement',inplace=True)
df_test['FireplaceQu'].fillna(value='No Fireplace',inplace=True)
df_test['GarageType'].fillna(value='No Garage',inplace=True)
df_test['GarageYrBlt'].fillna(value=0,inplace=True)
df_test['GarageFinish'].fillna(value='No Garage',inplace=True)
df_test['GarageQual'].fillna(value='No Garage',inplace=True)
df_test['GarageCond'].fillna(value='No Garage',inplace=True)
df_test['MasVnrType'].fillna(value='None',inplace=True)
df_test['MasVnrArea'].fillna(value=0.0,inplace=True)
df_test['PoolQC'].fillna(value='No Pool',inplace=True)
df_test['Fence'].fillna(value='No Fence',inplace=True)
df_test['MiscFeature'].fillna(value='None',inplace=True)
df_test['GarageCars'].fillna(value=0,inplace=True)
df_test['GarageArea'].fillna(value=0,inplace=True)
df_test['TotalBsmtSF'].fillna(value=0,inplace=True)
df_test['BsmtFinSF1'].fillna(value=0,inplace=True)

plt.figure(figsize=(20,8))
sns.heatmap(df_test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# In[8]:


df_test.isnull().sum().sort_values(ascending=False).head(15)


# In[9]:


Categorical_features = df_train.select_dtypes(include=['object'])
Numerical_features = df_train.select_dtypes(exclude=['object'])


# # **Looking at the data distribution**
# 
# Lets look if there is any clear correlation in the data

# In[10]:


#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# As predicting sale price of the houses is our aim lets look at areas where we see clear correlation

# In[11]:


corr = df_train.corr()
corr[corr['SalePrice']>0.3].index


# In[12]:


#scatterplot
sns.set()
cols=['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
       'WoodDeckSF', 'OpenPorchSF', 'SalePrice']
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
#cols=['SalePrice','MSSubClass','LotArea','YearBuilt','YrSold','1stFlrSF','2ndFlrSF','PoolArea','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
sns.pairplot(df_train[cols])
plt.show()


# The pair plot is a bit overwhelming so lets narrow it down and just look at the Sale price relation to the other data.

# In[13]:


f,ax=plt.subplots(3,6,figsize=(20,10))
k=0
for i in range(3):
    for j in range(6):
        sns.scatterplot(df_train[cols[k]],df_train['SalePrice'],ax=ax[i,j])
        plt.xlabel(cols[k])
        k=k+1
f.tight_layout()
plt.show()


# Overall quality of the property have a big impact on the sale price so lets see if there is a relationship.

# In[14]:


cols=['LotFrontage', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
       'WoodDeckSF', 'OpenPorchSF']

f,ax=plt.subplots(8,2,figsize=(20,30))
k=0
for i in range(8):
    for j in range(2):
        sns.scatterplot(df_train[cols[k]],df_train['SalePrice'],hue=df_train['OverallQual'],ax=ax[i,j])
        plt.legend(loc='best')
        k=k+1

f.tight_layout()
plt.show()


# There seems to some clear relationship between overall quality of the property, sale price other variables

# # **The Test train split**
# 
# This test train split of the training data will be used to validate all the models

# In[15]:


from sklearn.model_selection import train_test_split

from sklearn import metrics

feature=['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea',
       'WoodDeckSF', 'OpenPorchSF']

y=df_train['SalePrice']
X=df_train[feature]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8,random_state=42)


# # **Using Linear Regression**
# 
# Lets see if using linear regression on these relationships can give accurate predictions 

# In[16]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
preds = LR.predict(X_valid)

print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, preds))))


# In[17]:


plt.figure(figsize=(15,8))
plt.scatter(y_valid,preds)
plt.xlabel('Y_valid sales price')
plt.ylabel('Predicted Y sales price')
plt.show()


# Not that linear of a relation but its a good start

# In[18]:


# Fit the model to the training data
LR.fit(X, y)

# Generate test predictions
preds_test = LR.predict(df_test[feature])


# In[19]:


output_csv = pd.DataFrame({'Id': df_test.Id, 'SalePrice': preds_test})
output_csv.to_csv('linearRe.csv', index=False)
print("Your submission was successfully saved!")


# # **Using Random Forest**
# 
# We will compare different settings in the Random forest model to see which works the best

# In[20]:


from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=100,random_state=0)
model_2 = RandomForestRegressor(n_estimators=100,criterion='mae',random_state=0)
model_3 = RandomForestRegressor(n_estimators=200,min_samples_split=20,random_state=0)
model_4 = RandomForestRegressor(n_estimators=100,max_depth=7,random_state=0)

models = [model_1, model_2, model_3, model_4]


# In[21]:


# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    print(("Model %d:" % (i+1)))
    print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_v, preds))))
    return

for i in range(0, len(models)):
    mae = score_model(models[i])


# In[22]:


model_2.fit(X_train,y_train)

# Generate test predictions
preds = model_2.predict(X_valid[feature])

plt.figure(figsize=(15,8))
plt.scatter(y_valid,preds)
plt.xlabel('Y_valid sales price')
plt.ylabel('Predicted Y sales price')
plt.show()


# Pretty linear relation so we have headed in the right direction.

# In[23]:


# Fit the model to the training data
model_2.fit(X, y)

# Generate test predictions
preds_test = model_2.predict(df_test[feature])


# In[24]:


output_csv = pd.DataFrame({'Id': df_test.Id, 'SalePrice': preds_test})
output_csv.to_csv('randomforest.csv', index=False)
print("Your submission was successfully saved!")


# # **Gradient Tree Boosting**

# In[25]:


from sklearn.ensemble import GradientBoostingRegressor

GBR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01,max_depth=4, random_state=0, loss='ls')
GBR.fit(X_train,y_train)
preds = GBR.predict(X_valid)

print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, preds))))


# In[26]:


plt.figure(figsize=(15,8))
plt.scatter(y_valid,preds)
plt.xlabel('Y_valid sales price')
plt.ylabel('Predicted Y sales price')
plt.show()


# In[27]:


# Fit the model to the training data
GBR.fit(X, y)

# Generate test predictions
preds_test = GBR.predict(df_test[feature])

output_csv = pd.DataFrame({'Id': df_test.Id, 'SalePrice': preds_test})
output_csv.to_csv('GBR.csv', index=False)
print("Your submission was successfully saved!")

