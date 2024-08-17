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


train=pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')
test=pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')


# In[3]:


print(('Train Size:',train.shape))
print(('Test Size:',test.shape))


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


numeric_data=train.select_dtypes(exclude='object').drop(['SalePrice','Id'],axis=1).copy()
numeric_data.head()

categorical_data=train.select_dtypes(include='object')
categorical_data.head()


# In[7]:


#Distribution plot
import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20,30))
for i in range(len(numeric_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.distplot(numeric_data.iloc[:,i].dropna(), rug=True, hist=True, kde_kws={'bw':0.1})
    plt.xlabel(numeric_data.columns[i])
plt.tight_layout()
plt.show()


# In[8]:


#Boxplot(Univariate Analysis)
fig=plt.figure(figsize=(10,15))
for i in range(len(numeric_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=numeric_data.iloc[:,i])
plt.tight_layout()
plt.show()


# In[9]:


#Count plot (categorical, univariate analysis)
fig=plt.figure(figsize=(18,20))
for i,col in enumerate(categorical_data):
    fig.add_subplot(11,4,i+1)
    sns.countplot(train[col])
    plt.xlabel(col)
    plt.xticks(rotation=90)
plt.tight_layout(pad=1)
plt.show()


# In[10]:


#Scatterplot(bivariate)
fig=plt.figure(figsize=(20,30))
for i in range(len(numeric_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.scatterplot(numeric_data.iloc[:,i],train['SalePrice'])
plt.tight_layout()
plt.show()


# In[11]:


#Correlation
num=train.select_dtypes(exclude='object').drop('Id',axis=1)
numeric_correlation=num.corr()
plt.figure(figsize=(10,10))
plt.title('Correlation')
sns.heatmap(numeric_correlation>0.8, annot=True, square=True)


# In[12]:


print((numeric_correlation['SalePrice'].sort_values(ascending=False)))


# In[13]:


#dropping features due to high correlation
train.drop(['GarageYrBlt','TotRmsAbvGrd','GarageCars','1stFlrSF','YearBuilt'],axis=1,inplace=True)
test.drop(['GarageYrBlt','TotRmsAbvGrd','GarageCars','1stFlrSF','YearBuilt'],axis=1,inplace=True)


# In[14]:


#no linear relationship with SalePrice
train.drop(['MSSubClass','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','Fireplaces','MoSold','YrSold'],axis=1,inplace=True)
test.drop(['MSSubClass','OverallCond','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','Fireplaces','MoSold','YrSold'],axis=1,inplace=True)


# In[15]:


#how many missing values
train.isnull().mean().sort_values(ascending=False)


# In[16]:


#drop columns with high no. of null values
train.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
test.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)


# In[17]:


#drop categorical columns with mostly a single value in it, in this case more than 95%
remove_col=[]
categorical_data=train.select_dtypes(include='object').columns
for i in categorical_data:
    total_count=train[i].value_counts()
    zero_count=total_count.iloc[0]
    if zero_count/len(train)*100 > 95:
        remove_col.append(i)

print(remove_col)
train.drop(remove_col,axis=1,inplace=True)
test.drop(remove_col,axis=1,inplace=True)


# In[18]:


#drop numeric columns
remove_col=[]
numeric_data=train.select_dtypes(exclude='object').columns
for i in numeric_data:
    total_count=train[i].value_counts()
    zero_count=total_count.iloc[0]
    if zero_count/len(train)*100 > 95:
        remove_col.append(i)

print(remove_col)
train.drop(remove_col,axis=1,inplace=True)
test.drop(remove_col,axis=1,inplace=True)


# In[19]:


#Identifying outliers
numeric_data=train.select_dtypes(exclude='object').drop(['Id','SalePrice'],axis=1).copy()
fig=plt.figure(figsize=(10,15))
for i in range(len(numeric_data.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=numeric_data.iloc[:,i])
plt.tight_layout()
plt.show()


# In[20]:


#remove outliers
train=train.drop(train[train['LotFrontage']>200].index)
train=train.drop(train[train['LotArea']>100000].index)
train=train.drop(train[train['BsmtFinSF1']>4000].index)
train=train.drop(train[train['TotalBsmtSF']>4000].index)
train=train.drop(train[train['GrLivArea']>4000].index)
train=train.drop(train[train['EnclosedPorch']>400].index)


# In[21]:


#Missing Values
pd.DataFrame(test.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(51)


# In[22]:


test.head()


# In[23]:


num=train.select_dtypes(exclude='object').drop(['SalePrice'],axis=1).columns
cat=train.select_dtypes(include='object').columns


# In[24]:


#changing NA in numerical features to their mean
for i in train,test:
    for j in num:
        i[j]=i[j].fillna(i[j].mean())
        
train['MasVnrArea']=train['MasVnrArea'].fillna(0)
test['MasVnrArea']=test['MasVnrArea'].fillna(0)


# In[25]:


#changing NA in categorical features to 'None'
ordinal_features=['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1','BsmtFinType2', 'Neighborhood', 'BldgType', 'HouseStyle', 'MasVnrType', 'FireplaceQu']
categorical_features=['MSZoning','LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Condition1', 'RoofStyle','Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'ExterQual', 'ExterCond','Foundation', 'HeatingQC', 'CentralAir', 'PavedDrive', 'SaleCondition']

for i in train,test:
    for j in ordinal_features:
        i[j]=i[j].fillna('None')

for i in train,test:
    for j in categorical_features:
        i[j]=i[j].fillna(i[j].mode()[0])


# In[26]:


pd.DataFrame(test.isnull().sum(),columns=['Sum']).sort_values(by='Sum',ascending=False).head(10)


# In[27]:


train.columns


# In[28]:


#Feature Engineering

train['TotalLot'] = train['LotFrontage'] + train['LotArea']
train['TotalBsmtFin'] = train['BsmtFinSF1'] + train['BsmtFinSF2']
train['TotalSF'] = train['TotalBsmtSF'] + train['2ndFlrSF']
train['TotalPorch'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['ScreenPorch']

test['TotalLot'] = test['LotFrontage'] + test['LotArea']
test['TotalBsmtFin'] = test['BsmtFinSF1'] + test['BsmtFinSF2']
test['TotalSF'] = test['TotalBsmtSF'] + test['2ndFlrSF']
test['TotalPorch'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['ScreenPorch']


# In[29]:


#changing columns to binary to check if that feature is there in the house, 1(yes), 0(no)
bin_columns = ['MasVnrArea','TotalBsmtFin','TotalBsmtSF','2ndFlrSF','WoodDeckSF','TotalPorch']
for i in bin_columns:
    col=i+'_bin'
    train[col] = train[i].apply(lambda x: 1 if x>0 else 0)
    test[col] = test[i].apply(lambda x: 1 if x>0 else 0)


# In[30]:


train.head()


# In[31]:


plt.figure(figsize=(10,6))
plt.title("Distrubution of SalePrice")
dist = sns.distplot(train['SalePrice'],norm_hist=False)


# This gives a skewed result, so we apply log on that

# In[32]:


plt.figure(figsize=(10,6))
plt.title("Distrubution of SalePrice")
dist = sns.distplot(np.log(train['SalePrice']),norm_hist=False)


# In[33]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor


# In[34]:


from sklearn.model_selection import train_test_split

x = train.drop(['SalePrice'], axis=1) 
y = np.log1p(train['SalePrice'])
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

categorical_cols = [cname for cname in x.columns if
                    x[cname].nunique() <= 30 and
                    x[cname].dtype == "object"] 
                


numerical_cols = [cname for cname in x.columns if
                 x[cname].dtype in ['int64','float64']]


my_cols = numerical_cols + categorical_cols
X_train = X_train[my_cols].copy()
X_val = X_val[my_cols].copy()
X_test = test[my_cols].copy()


# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

num_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='constant'))
    ])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),       
        ('cat',cat_transformer,categorical_cols),
        ])


# In[36]:


# Reversing log-transform on y
def inv_y(transformed_y):
    return np.exp(transformed_y)

n_folds = 10


# In[37]:


'''# XGBoost
model = XGBRegressor(learning_rate=0.01, n_estimators=3460, max_depth=3, min_child_weight=0,gamma=0, subsample=0.7,colsample_bytree=0.7,objective='reg:squarederror', nthread=-1,scale_pos_weight=1, seed=27, reg_alpha=0.00006)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_val)
print('XGBoost: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_val))))


# Lasso  
from sklearn.linear_model import LassoCV

model = LassoCV(max_iter=1e7,  random_state=14, cv=n_folds)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_val)
print('Lasso: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_val))))
  
      
      
# GradientBoosting   
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
clf.fit(X_train, y_train)
predict = clf.predict(X_val)
print('Gradient: ' + str(mean_absolute_error(inv_y(predict), inv_y(y_val))))'''


# In[38]:


model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)

final_model = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)


# In[39]:


output = pd.DataFrame({'Id': X_test.Id,
                       'SalePrice': inv_y(final_predictions)})

output.to_csv('submission.csv', index=False)


# In[ ]:




