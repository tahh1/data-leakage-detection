#!/usr/bin/env python
# coding: utf-8

# # Final Project for Data Science
# Yanjun Tan

# # Overview
# 

# * For this final project, I am planning to use various **regression methods** (simple regression, multiple regression, KNN regression...) and try to predict the house prices based on those regression algorithm.
# 
# * We are using **House Price Datasets** which contains thousands of records. I will build several prediction models based on current dataset, use the model I built to make prediction, and finally discuss pros and cons of each. 
# 
# * Regression algorithm could help us to have a deeper insight of data, as well as defining the relation between the response and explanatory variables.
# 
# * I will start with a very simple model (1 to 1) and continue with more complex ones (mupltiple to 1, KNN...) after visualizing some features and a data mining process. 
# 
# * The final purpost of this project is try to find the best prediction model for this dataset.
# 
# 

# ![](https://www.onthemarket.com/content/wp-content/uploads/2018/01/Housepricepredictions2018lead.jpg)

# # 1. Preparation

# > # 1.1 Import Modules

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
import numpy as np


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer

#models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#validation libraries
from sklearn.model_selection import KFold, StratifiedKFold
from IPython.display import display
from sklearn import metrics


from mpl_toolkits.mplot3d import Axes3D
import folium
from folium.plugins import HeatMap

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet,Lasso,SGDRegressor,Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import LabelEncoder,Imputer,OneHotEncoder,RobustScaler,StandardScaler,Imputer

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

from IPython.display import HTML, display

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# > > 

# > # 1.2 Load and Read Dataset

# In[2]:


housetrain= pd.read_csv("../input/train.csv")
housetest = pd.read_csv("../input/test.csv")
housetrain.head()


# In[3]:


housetest.head()


# # 2. Data Preprocessing
# > # 2.1 Data Cleansing 

# * **Check for DataFrame Info**

# In[4]:


df=pd.DataFrame(housetrain)
df.info()


# In[5]:


test=pd.DataFrame(housetest)
test.info()


# * **Check for Null Value Counts**

# In[6]:


missing_val_count_by_column = (df.isnull().sum())
print((missing_val_count_by_column[missing_val_count_by_column > 0]))


# In[7]:


print((missing_val_count_by_column[missing_val_count_by_column > 0].plot(kind='bar')))


# * **Dealing with Missing Value**

# 1. Fill the missing value with proper content
# 1. Following variable description to fill the null value in different columns. 

# 
# 

# # Remove Null

# In[8]:


# fill up MSZoning with the mode value
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# from the data description file, NA = No Alley Access
df['Alley'].fillna(0, inplace=True)

# fill up NA values with mode
df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])

# since both Exterior1st and 2nd only has 2 missing value, substitute with mode
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

# fill up MasVnrType with the mode value
df["MasVnrType"] = df["MasVnrType"].fillna(df['MasVnrType'].mode()[0])
df["MasVnrArea"] = df["MasVnrArea"].fillna(df['MasVnrArea'].mode()[0])

# for these columns, NA = No Basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')
    
# for these columns, NA is likely to be 0 due to no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col] = df[col].fillna(0)
    
# substitue NA value here with mode
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# substitute NA value with mode
df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

# if no value, assume Typ, typical is also mode value
df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])

# NA = No Fireplace
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

# for these columns, NA = No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[col] = df[col].fillna('None')
    
# as there is no garage, NA value for this column is set to zero
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(0)
    
# NA = no pool
df['PoolQC'] = df['PoolQC'].fillna('None')

# NA = no fence
df['Fence'] = df['Fence'].fillna('None')

#Misc Feature, NA = None
df['MiscFeature'] = df['MiscFeature'].fillna('None')

#sale type, only have 1 NA value. substitute it with mode value
df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

# checking for any null value left
df.isnull().sum().sum()


# In[9]:


# fill up MSZoning with the mode value
test['MSZoning'] = test['MSZoning'].fillna(test['MSZoning'].mode()[0])

# LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
test["LotFrontage"] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

# from the data description file, NA = No Alley Access
test["Alley"] = test["Alley"].fillna(0)

# fill up NA values with mode
test['Utilities'] = test['Utilities'].fillna(test['Utilities'].mode()[0])

# since both Exterior1st and 2nd only has 2 missing value, substitute with mode
test['Exterior1st'] = test['Exterior1st'].fillna(test['Exterior1st'].mode()[0])
test['Exterior2nd'] = test['Exterior2nd'].fillna(test['Exterior2nd'].mode()[0])

# fill up MasVnrType with the mode value
test["MasVnrType"] = test["MasVnrType"].fillna(df['MasVnrType'].mode()[0])
test["MasVnrArea"] = test["MasVnrArea"].fillna(df['MasVnrArea'].mode()[0])

# for these columns, NA = No Basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    test[col] = test[col].fillna('None')
    
# for these columns, NA is likely to be 0 due to no basement
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    test[col] = test[col].fillna(0)
    
# substitue NA value here with mode
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])

# substitute NA value with mode
test['KitchenQual'] = test['KitchenQual'].fillna(test['KitchenQual'].mode()[0])

# if no value, assume Typ, typical is also mode value
test['Functional'] = test['Functional'].fillna(test['Functional'].mode()[0])

# NA = No Fireplace
test['FireplaceQu'] = test['FireplaceQu'].fillna('None')

# for these columns, NA = No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    test[col] = test[col].fillna('None')
    
# as there is no garage, NA value for this column is set to zero
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    test[col] = test[col].fillna(0)
    
# NA = no pool
test['PoolQC'] = test['PoolQC'].fillna('None')

# NA = no fence
test['Fence'] = test['Fence'].fillna('None')

#Misc Feature, NA = None
test['MiscFeature'] = test['MiscFeature'].fillna('None')

#sale type, only have 1 NA value. substitute it with mode value
test['SaleType'] = test['SaleType'].fillna(test['SaleType'].mode()[0])

# checking for any null value left
test.isnull().sum().sum()


# In[10]:





# In[10]:





# In[10]:





# In[10]:


# feature extraction



# In[11]:


numeric_cols = [x for x in df.columns if ('Area' in x) | ('SF' in x)] + ['SalePrice','LotFrontage','MiscVal','EnclosedPorch','3SsnPorch','ScreenPorch','OverallQual','OverallCond','YearBuilt']

for col in numeric_cols:
    df[col] = df[col].astype(float)
numeric_cols


# In[12]:


categorical_cols = [x for x in df.columns if x not in numeric_cols]

for col in categorical_cols:
    df[col] = df[col].astype('category')
    
categorical_cols


# # Convert Continuous to Categorical

# In[13]:


df['above_200k'] = df['SalePrice'].map(lambda x : 1 if x > 200000 else 0) 
df['above_200k'] = df['above_200k'].astype('category')

df.loc[df['SalePrice']>200000,'above_200k'] = 1
df.loc[df['SalePrice']<=200000,'above_200k'] = 0
df['above_200k'] = df['above_200k'].astype('category')


# In[14]:


df['LivArea_Total'] = df['GrLivArea'] + df['GarageArea'] + df['PoolArea']
df[['LivArea_Total','GrLivArea','GarageArea','PoolArea']].head()


# In[15]:


## concatenating two different fields together in the same row
df['Lot_desc'] = df.apply(lambda val : val['MSZoning'] + val['LotShape'], axis=1)
df[['Lot_desc','MSZoning','LotShape']].head()


# # Scale Fields

# In[16]:


from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer

df['LotArea_norm'] = df['LotArea']

ss = StandardScaler()
mas = MaxAbsScaler()
qs = QuantileTransformer()

df['LotArea_norm'] = ss.fit_transform(df[['LotArea']])
df['LotArea_mas'] = mas.fit_transform(df[['LotArea']])
df['LotArea_qs'] = qs.fit_transform(df[['LotArea']])


df[['LotArea_norm','LotArea_mas','LotArea_qs', 'LotArea']].head(5)


# # Words/Labels as Features

# In[17]:


small_df = df[['MSZoning','SalePrice']].copy()
small_df['MSZoning'] = small_df['MSZoning'].astype('category')
small_df.head()


# In[18]:


pd.get_dummies(small_df).head(5)


# In[19]:


small_df = df[['MSSubClass','SalePrice']].copy()
small_df['MSSubClass'] = small_df['MSSubClass'].astype('category')
small_df.head()


# In[20]:


le = LabelEncoder()
trf_MSSubClass = le.fit_transform(small_df['MSSubClass'])
trf_MSSubClass


# In[21]:


le.classes_


# In[22]:


le.inverse_transform(trf_MSSubClass)


# In[23]:


feature_cols = [col for col in df.columns if 'Price' not in col]


# In[24]:


print(feature_cols)


# In[25]:


df['LogSalePrice'] = np.log(df['SalePrice'])

y = df['LogSalePrice']
X = df[feature_cols]
print((X.head(2),'\n\n', X.head(2)))


# In[26]:


X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)
print((X_train.shape, X_valid.shape, y_train.shape, y_valid.shape))


# In[27]:


X_numerical = pd.get_dummies(X)
X_numerical.head(5)


# In[28]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from boruta import boruta_py
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from pandas import read_csv


def greedy_elim(df):

    # do feature selection using boruta
    X = df[[x for x in df.columns if x!='SalePrice']]
    y = df['SalePrice']
    #model = RandomForestRegressor(n_estimators=50)
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05)
    # 150 features seems to be the best at the moment. Why this is is unclear.
    feat_selector = RFE(estimator=model, step=1, n_features_to_select=150)

    # find all relevant features
    feat_selector.fit_transform(X.as_matrix(), y.as_matrix())

    # check selected features
    features_bool = np.array(feat_selector.support_)
    features = np.array(X.columns)
    result = features[features_bool]
    #print(result)

    # check ranking of features
    features_rank = feat_selector.ranking_
    #print(features_rank)
    rank = features_rank[features_bool]
    #print(rank)

    print(result) 
    print(rank)







# In[29]:





# > # 2.2 Data Observation

# * **Calculate Statistics**

# In[29]:


# Minimum price of the data
minimum_price = np.amin(df.SalePrice)

# Maximum price of the data
maximum_price = np.amax(df.SalePrice)


# Mean price of the data
mean_price = np.mean(df.SalePrice)


# Median price of the data
median_price = np.median(df.SalePrice)


# Standard deviation of prices of the data
std_price = np.std(df.SalePrice)

    
# Show the calculated statistics
print("Zillow Housing Price Dataset:\n")
print(("Minimum price: ${}".format(minimum_price))) 
print(("Maximum price: ${}".format(maximum_price)))
print(("Mean price: ${}".format(mean_price)))
print(("Median price ${}".format(median_price)))
print(("Standard deviation of prices: ${}".format(std_price)))


#  > **Visualize Sales Price** (房价呈正态分布)

# In[30]:


sns.distplot(df['SalePrice'])


# *  **House Age / Remodel Age Distribution When the Houses Were Sold (Histograms)**

# In[31]:


newhouse_dm=df.copy()
newhouse_dm.head()


# In[32]:


# add the age of the buildings when the houses were sold as a new column
newhouse_dm['Age']=newhouse_dm['YrSold'].astype(int)-newhouse_dm['YearBuilt'].astype(int)

# partition the age into bins
bins = [-2,0,5,10,25,50,75,100,100000]
labels = ['<1','1-5','6-10','11-25','26-50','51-75','76-100','>100']
newhouse_dm['age_binned'] = pd.cut(newhouse_dm['Age'].astype(int), bins=bins, labels=labels)


# add the age of the renovation when the houses were sold as a new column
newhouse_dm['age_remodel']=0
newhouse_dm['age_remodel']=newhouse_dm['YrSold'][newhouse_dm['YearRemodAdd']!=0].astype(int)-newhouse_dm['YearRemodAdd'].astype(int)[newhouse_dm['YearRemodAdd']!=0].astype(int)
newhouse_dm['age_remodel'][newhouse_dm['age_remodel'].isnull()]=0


# histograms for the binned columns
f, axes = plt.subplots(1,2,figsize=(25,5))
p1=sns.countplot(newhouse_dm['age_binned'],ax=axes[0])

# partition the age_remodel into bins
bins = [-2,0,5,10,25,50,75,100000]
labels = ['<1','1-5','6-10','11-25','26-50','51-75','>75']
newhouse_dm['age_remodel_binned'] = pd.cut(newhouse_dm['age_remodel'], bins=bins, labels=labels)


for p in p1.patches:
    height = p.get_height()
    p1.text(p.get_x()+p.get_width()/2,height + 50,height,ha="center")   

p2=sns.countplot(newhouse_dm['age_remodel_binned'],ax=axes[1])
sns.despine(left=True, bottom=True)
for p in p2.patches:
    height = p.get_height()
    p2.text(p.get_x()+p.get_width()/2,height + 200,height,ha="center")
    
axes[0].set(xlabel='Age')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Remodel Age');

# transform the factor values to be able to use in the model
newhouse_dm = pd.get_dummies(newhouse_dm, columns=['age_binned'])


# # RFE

# In[33]:





# In[33]:





# # 3. Feature Observation & Selection

# > # ** 3.1 Observe Different Feature's Influence on House Price**

# * There are totally 36 features in our dataset.
# * It is really too much to put in a model.
# * Sometimes it may cause overfitting and worser results when we want to predict values for a new dataset. 
# * We want to remove those variables if they can't improve our model.
# 
# 
# * Another important thing is correlation, if there is very high correlation between two features, keeping both of them is not a good idea most of the time. For instance, **1stSquareFoot** and **2ndSquareFoot** are highly correlated. 
# * This can be estimated when you look at the definitions at the dataset and checked to be sure by looking at the correlation matrix. 
# * However, this does not mean that you must remove one of the highly correlated features.
# * For instance: BedroomAbvGr and TotRmsAbvGrd. They are highly correlated but I do not think that the relation among them is the same as the relation between 1stSquareFoot and 2nd SquareFoot.

#  > * **Categorical Varialbes **

# In[33]:


# CentralAir
var = 'CentralAir'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=350000);


# In[34]:


# OverallQual
var = 'OverallQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=500000);


# In[35]:


# YearBuilt
var = 'YearBuilt'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))


# In[36]:


var = 'Utilities'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=350000);


# In[37]:


var = 'SaleCondition'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=350000);


# In[38]:


# Neighborhood
var = 'Neighborhood'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# * **Numerical Variables**

# **Lot area seems not much relationship with price**

# In[39]:


#GrLivArea

var  = 'GrLivArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[40]:


# GarageArea 

var  = 'GarageArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[41]:


# LotArea

var  = 'LotArea'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# In[42]:


# BedroomAbvGr
var = 'BedroomAbvGr'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[43]:


#TotalBsmtSF

var  = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


# > # 3.2 Checking Out the Correlation Among Explanatory Variables
# 
# **LotArea, BedroomAbvGr, TotRmsAbvGrd and GrLivingArea seem to have a high influence in price
# **

# In supervised learning, correlated features in general don't improve models (although it depends on the specifics of the problem like the number of variables and the degree of correlation), **but they affect specific models in different ways and to varying extents**:
# 
# * **For linear models** (e.g., linear regression or logistic regression), multicolinearity can yield solutions that are wildly varying and possibly numerically unstable.
# 
# * **Random forests** can be good at detecting interactions between different features, but highly correlated features can mask these interactions.
# 
# More generally, this can be viewed as a special case of Occam's razor. A simpler model is preferable, and, in some sense, a model with fewer features is simpler. The concept of minimum description length makes this more precise.

# > * **Pearson Correlation Matrix**

# In[44]:


features = ['SalePrice','MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',
            'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
            '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
            'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','BsmtHalfBath','GarageYrBlt',
            'GarageCars','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
            'MiscVal','MoSold','GarageArea']

mask = np.zeros_like(df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# **High Correlation Features with House Price:
# **
# * OverallQual：总评价
# * YearBuilt：建造年份
# * YearRemodAdd
# * ToatlBsmtSF：地下室面积
# * 1stFlrSF：一楼面积
# * GrLiveArea：生活区面积
# * FullBath: 全浴室数
# * TotRmsAbvGrd：总房间数（不包括浴室）
# * GarageCars：车库可容纳车辆数
# * GarageArea：车库面积

# In[45]:


from sklearn import preprocessing

f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    df[x] = label.fit_transform(df[x])
corrmat = df.corr()
f, ax = plt.subplots(figsize=(20, 9))

k  = 11 # 关系矩阵中将显示10个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 10}, cmap='PiYG',yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[46]:


sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']
sns.pairplot(df[cols], size = 2.5)
plt.show()


# Lasso Regresson

# # 4. Create Evaluation Table

# In[47]:


evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Mean Squared Error (MSE)':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})


# > Defining a Function to Calculate the Adjusted $R^2$

# The R-squared increases along with the number of features increases. 
# Because of this, sometimes a more robust evaluator is preferred to compare the performance between different models. 
# This evaluater is called **Adjusted R-squared** and it only increases, if the addition of the variable reduces the MSE. 
# 
# The definition of the** adjusted $R^2$** is:
# $\bar{R^{2}}=R^{2}-\frac{k-1}{n-k}(1-R^{2})$
# 

# In[48]:


def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)


# # RFE Feature Selection

# In[49]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets

names = df[['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']]

target=df[['SalePrice']]

svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 3)
rfe = rfe.fit(names, target)
# print summaries for the selection of attributes
list(names)
print((rfe.support_))
print((rfe.ranking_))


# In[50]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets

names = df[['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']]

target=df[['SalePrice']]

svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 4)
rfe = rfe.fit(names, target)
# print summaries for the selection of attributes
list(names)
print((rfe.support_))
print((rfe.ranking_))


# In[51]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets

names = df[['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','GarageArea','1stFlrSF']]

target=df[['SalePrice']]

svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 5)
rfe = rfe.fit(names, target)
# print summaries for the selection of attributes
list(names)
print((rfe.support_))
print((rfe.ranking_))


# # 5. Build Simple Linear Regression Model - 1 to 1

# > **<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
#   <msub>
#     <mi>y</mi>
#     <mi>i</mi>
#   </msub>
#   <mo>=</mo>
#   <mi>&#x03B1;<!-- α --></mi>
#   <mo>+</mo>
#   <mi>&#x03B2;<!-- β --></mi>
#   <msub>
#     <mi>x</mi>
#     <mi>i</mi>
#   </msub>
#   <mo>+</mo>
#   <msub>
#     <mi>&#x03F5;<!-- ϵ --></mi>
#     <mi>i</mi>
#   </msub>
# </math> **

# Where:
# 
# y = dependent variable
# 
# β = regression coefficient
# 
# α = intercept (expected mean value of housing prices when our independent variable is zero)
# 
# x = predictor (or independent) variable used to predict Y
# 
# ϵ = the error term, which accounts for the randomness that our model can't explain.
# 

# * In statistics, linear regression is a linear approach to modelling the relationship between a **scalar response** and **one or more explanatory variables**. 
# * **Simple linear regression** uses a single predictor variable to explain a dependent variable. 
# * The case of one explanatory variable is called **simple linear regression**.
# * In the first section, we want to use one explanatory variable to predict house price.

# > # 5.1 Model 1 - Simple Linear Regression - Lot Size v.s. Price
# 

# In[52]:


#####################################  Simple Linear Regression - Lot Size v.s. Price  ##################################### 

import math
#Split data into Train and Test (60%/40%)
train_data,test_data = train_test_split(df,train_size = 0.6,random_state=3)

#Using Linear Regression Model and Train Model with Training Subset
lr = linear_model.LinearRegression()
X_train = np.array(train_data['LotArea'], dtype=pd.Series).reshape(-1,1)
y_train = np.array(train_data['SalePrice'], dtype=pd.Series)

#Fitting Model 1 to Training Data
lr.fit(X_train,y_train)

X_test = np.array(test_data['LotArea'], dtype=pd.Series).reshape(-1,1)
y_test = np.array(test_data['SalePrice'], dtype=pd.Series)

pred = lr.predict(X_test)
msesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse=(math.sqrt(msesm))
rtrsm = float(format(lr.score(X_train, y_train),'.3f'))
rtesm = float(format(lr.score(X_test, y_test),'.3f'))
cv = float(format(cross_val_score(lr,df[['LotArea']],df['SalePrice'],cv=5).mean(),'.3f'))

print(("Average Price for Test Data: {:.3f}".format(y_test.mean())))
print(('Intercept: {}'.format(lr.intercept_)))
print(('Coefficient: {}'.format(lr.coef_)))

r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression - Lot Size v.s. Price','-',msesm,rmse,rtrsm,'-',rtesm,'-',cv]
evaluation


# In[53]:





# > **Coefficient Means?**

# * For an increase of 1 square feet in Lot Area Size,
# * The house price will go up by $2.5 on average

# > **Root Mean Squared Error of Model 1**

# * Root Mean Squared Error (MSE), which is a commonly used metric to evaluate regression models based on the test subset.
# * For this model, we get a root mean squared error of $12118.05 when predicting a price for a house, which is really high.
# * The reason why the MSE value is so high is because we’re only using one feature in our model, and it could be greatly improved by adding more features.
# * We can also see that we’re omitting relevant variables by looking at the R squared coefficient: 54.4%. 
# * This means that our model is only able to explain 54.4% of the variability in house prices.

# > **Then we want to use this model to predict the house price of a house with 1000 sqr feet lot area.**

# In[53]:


lr.fit(X_train,y_train)
lr.predict([[1000]])


# In[54]:


lr.score(X_test,y_test)


# * 在对回归模型进行校验时，判断系数R²也称拟合优度或决定系数，即相关系数R的平方，用于表示拟合得到的模型能解释因变量变化的百分比
# * R²越接近1，表示回归模型拟合效果越好。

# > **See Model in Axis**

# In[55]:


sns.set(style="white", font_scale=1)

plt.figure(figsize=(10,9))

plt.scatter(X_train,y_train,color='purple',label="Data", alpha=.1)
plt.plot(X_train,lr.predict(X_train),color="red",label="Predicted Regression Line")

plt.title("Simple Regression House Price Predict Model 1 - Train Set",fontsize=20)
plt.xlabel("LotArea (sqft)", fontsize=15)
plt.ylabel("SalePrice ($)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# In[56]:


sns.set(style="white", font_scale=1)

plt.figure(figsize=(10,9))

plt.scatter(X_test,y_test,color='purple',label="Data", alpha=.1)
plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")

plt.title("Simple Regression House Price Predict Model 1 - Test Set",fontsize=20)
plt.xlabel("LotArea (sqft)", fontsize=15)
plt.ylabel("SalePrice ($)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# > # 5.2 Model 2 - Simple Linear Regression - BedroomAbvGr v.s. Price
# 

# In[57]:


#############################################  BedroomAbvGr v.s. Price ###############################################################

train_data_1,test_data_1 = train_test_split(df,train_size = 0.6,random_state=3)

lr = linear_model.LinearRegression()
X_train_1 = np.array(train_data_1['BedroomAbvGr'], dtype=pd.Series).reshape(-1,1)
y_train_1 = np.array(train_data_1['SalePrice'], dtype=pd.Series)
lr.fit(X_train_1,y_train_1)

X_test_1 = np.array(test_data_1['BedroomAbvGr'], dtype=pd.Series).reshape(-1,1)
y_test_1 = np.array(test_data_1['SalePrice'], dtype=pd.Series)

pred = lr.predict(X_test_1)
msesm = float(format(np.sqrt(metrics.mean_squared_error(y_test_1,pred)),'.3f'))
rmse=(math.sqrt(msesm))
rtrsm = float(format(lr.score(X_train_1, y_train_1),'.3f'))
rtesm = float(format(lr.score(X_test_1, y_test_1),'.3f'))
cv = float(format(cross_val_score(lr,df[['LotArea']],df['SalePrice'],cv=5).mean(),'.3f'))

print(("Average Price for Test Data: {:.3f}".format(y_test.mean())))
print(('Intercept: {}'.format(lr.intercept_)))
print(('Coefficient: {}'.format(lr.coef_)))

r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression - Bedroom Above Grade v.s. Price','-',msesm,rmse,rtrsm,'-',rtesm,'-',cv]
evaluation


# * **For an increase of 1 rooms in Total Bedroom Numbers,**
# * **The house price will go up by $15708 on average**.

# **Root Mean Squared Error of Model 2**

# * **RMSE (Square root of MSE) = √ (MSE)**
# * **For this model, we get a root mean squared error of $12118.050410075917 when predicting a price for a house, which is still really high.** 
# * **We can also see that we’re omitting relevant variables by looking at the R squared coefficient: 56.2%. **
# * **This means that our model is only able to explain 56.2% of the variability in house prices.**

# **See Model in Axis**

# In[58]:


sns.set(style="white", font_scale=1)

plt.figure(figsize=(10,9))

plt.scatter(X_train_1,y_train_1,color='darkgreen',label="Data", alpha=.1)
plt.plot(X_train_1,lr.predict(X_train_1),color="red",label="Predicted Regression Line")

plt.title("Simple Regression House Price Predict Model 2 - Train Set",fontsize=20)
plt.xlabel("BedroomAbvGr", fontsize=15)
plt.ylabel("SalePrice ($)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# In[59]:


sns.set(style="white", font_scale=1)

plt.figure(figsize=(10,6))

plt.scatter(X_test_1,y_test_1,color='darkgreen',label="Data", alpha=.1)
plt.plot(X_test_1,lr.predict(X_test_1),color="red",label="Predicted Regression Line")

plt.title("Simple Regression House Price Predict Model 2 - Test Set",fontsize=20)
plt.xlabel("BedroomAbvGr", fontsize=15)
plt.ylabel("SalePrice ($)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# Then we want to predict the price for a house have 6 bedrooms.
# Then the predicted house price for 6 bedrooms based on this model is **$228334.2550356**
# 

# In[60]:


lr.fit(X_train_1,y_train_1)
lr.predict([[6]])


# > # 5.3 Model 3 - Simple Linear Regression - GrLivArea v.s. Price

# In[61]:


#############################################  GrLivArea v.s. Price ###############################################################

train_data_2,test_data_2 = train_test_split(df,train_size = 0.6,random_state=3)

lr = linear_model.LinearRegression()
X_train_2 = np.array(train_data_2['GrLivArea'], dtype=pd.Series).reshape(-1,1)
y_train_2 = np.array(train_data_2['SalePrice'], dtype=pd.Series)
lr.fit(X_train_2,y_train_2)

X_test_2 = np.array(test_data_2['GrLivArea'], dtype=pd.Series).reshape(-1,1)
y_test_2= np.array(test_data_2['SalePrice'], dtype=pd.Series)

pred = lr.predict(X_test_2)
msesm = float(format(np.sqrt(metrics.mean_squared_error(y_test_2,pred)),'.3f'))
rmse=(math.sqrt(msesm))
rtrsm = float(format(lr.score(X_train_2, y_train_2),'.3f'))
rtesm = float(format(lr.score(X_test_2, y_test_2),'.3f'))
cv = float(format(cross_val_score(lr,df[['LotArea']],df['SalePrice'],cv=5).mean(),'.3f'))

print(("Average Price for Test Data: {:.3f}".format(y_test.mean())))
print(('Intercept: {}'.format(lr.intercept_)))
print(('Coefficient: {}'.format(lr.coef_)))

r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression - Total Living Room Area v.s. Price','-',msesm,rmse,rtrsm,'-',rtesm,'-',cv]
evaluation


# > **From the evaluation, we know that:**
# * For an increase of 1 sqr foot in Total Living Room Area,
# * The house price will go up by $18.94478276 on average.

# > **Prediction Based on the Model
# **
# * Now we want to use this model to predict the price for a house with 1200 sqr feet living room.
# * We can know that the predicted price for the house is $173822.63535448

# In[62]:


lr.fit(X_train_2,y_train_2)
lr.predict([[1200]])


# > **Model in Axis**

# In[63]:


sns.set(style="white", font_scale=1)

plt.figure(figsize=(10,9))

plt.scatter(X_train_2,y_train_2,color='black',label="Data", alpha=.1)
plt.plot(X_train_2,lr.predict(X_train_2),color="red",label="Predicted Regression Line")

plt.title("Simple Regression House Price Predict Model 3 - Train Set",fontsize=20)
plt.xlabel("GrLivArea", fontsize=15)
plt.ylabel("SalePrice ($)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# In[64]:


sns.set(style="white", font_scale=1)

plt.figure(figsize=(10,6))

plt.scatter(X_test_2,y_test_2,color='black',label="Data", alpha=.1)
plt.plot(X_test_2,lr.predict(X_test_2),color="red",label="Predicted Regression Line")

plt.title("Simple Regression House Price Predict Model 3 - Test Set")
plt.xlabel("GrLivArea", fontsize=15)
plt.ylabel("SalePrice ($)", fontsize=15)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


# # 5.4 Conclusion

# * **Root Mean Squared Error (RMSE),** which is a commonly used metric to evaluate regression models based on the test subset.
# * **For this model, the root mean squared error we got when predicting a price for a house is really high.** 
# * The reason why the RMSE value is so high is because **we’re only using one feature in our model, and it could be greatly improved by adding more features.**

# # 6. Build Multiple Linear Regression Model - Many to 1

# > **<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
#   <mi>Y</mi>
#   <mo>=</mo>
#   <msub>
#     <mi>&#x03B2;<!-- β --></mi>
#     <mn>0</mn>
#   </msub>
#   <mo>+</mo>
#   <msub>
#     <mi>&#x03B2;<!-- β --></mi>
#     <mn>1</mn>
#   </msub>
#   <msub>
#     <mi>x</mi>
#     <mn>1</mn>
#   </msub>
#   <mo>+</mo>
#   <msub>
#     <mi>&#x03B2;<!-- β --></mi>
#     <mn>2</mn>
#   </msub>
#   <msub>
#     <mi>x</mi>
#     <mn>2</mn>
#   </msub>
#   <mo>+</mo>
#   <mo>.</mo>
#   <mo>.</mo>
#   <mo>.</mo>
#   <mo>+</mo>
#   <msub>
#     <mi>&#x03B2;<!-- β --></mi>
#     <mi>k</mi>
#   </msub>
#   <msub>
#     <mi>x</mi>
#     <mi>k</mi>
#   </msub>
#   <mo>+</mo>
#   <mi>&#x03F5;<!-- ϵ --></mi>
# </math>**

# * In previous sections, I used simple linear regression and found a poor fit. 
# * To get a clearer picture of what influences housing prices as well as to improve the prediction model, I'm planing to testing more variables to analyze the regression results 
# * to see which combinations of predictor variables satisfy OLS assumptions the most.
# * When it contains more than one feature in a linear regression as variables, then it turns to be multiple linear regression. 
# * In the following section, I'm going to create some more complex models.

# In[65]:


df.info()


# > **Remove All Object Variables from Features**

# In[66]:


features=['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'MoSold',
 'YrSold',
 'SalePrice']



# In[67]:


print(('this many columns:%d ' % len(df.columns)))
df.columns


# In[68]:


feature_cols = [col for col in df.columns if 'Price' not in col]


# In[69]:


print (feature_cols)


# In[70]:


X_numerical = pd.get_dummies(X)
X_numerical.head(5)


# In[71]:





# # 6.1 Multiple Linear Regression Model 1 - All Features

# In[71]:


train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

complex_model_1 = linear_model.LinearRegression()
complex_model_1.fit(train_data_dm[features],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_1.intercept_)))
print(('Coefficients: {}'.format(complex_model_1.coef_)))

pred = complex_model_1.predict(test_data_dm[features])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rmse=(math.sqrt(msecm))
rtrcm = float(format(complex_model_1.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_1.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm = float(format(complex_model_1.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_1.score(test_data_dm[features],test_data['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))
cv = float(format(cross_val_score(complex_model_1,df[features],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 1 - All Features','All Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# In[72]:





# # 6.2 Multiple Linear Regression Model 2 - Selected Features
# **'YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea' v.s. House Price**

# * OverallQual：总评价
# * YearBuilt：建造年份
# * YearRemodAdd:重修年份
# * TotBsmtSF：地下室面积
# * 1stFlrSF：一楼面积
# * GrLiveArea：生活区面积
# * FullBath: 全浴室数
# * TotRmsAbvGrd：总房间数（不包括浴室）
# * GarageCars：车库可容纳车辆数
# * GarageArea：车库面积

# In[72]:


############################################# Features 2 ##################################################

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features_2=['YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']

complex_model_2 = linear_model.LinearRegression()
complex_model_2.fit(train_data_dm[features_2],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_2.intercept_)))
print(('Coefficients: {}'.format(complex_model_2.coef_)))


# In[73]:


pred = complex_model_2.predict(test_data_dm[features_2])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rmse=(math.sqrt(msecm))
rtrcm = float(format(complex_model_2.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_2.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))
rtecm = float(format(complex_model_2.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_2.score(test_data_dm[features_2],test_data['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))
cv = float(format(cross_val_score(complex_model_2,df[features_2],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 2 - 9 Features v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# In[74]:


features_2=['YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']


# * **Using Model_2 to predict a house whose Overall Quality Score is 10, and Overall Condition Score is 8**

# In[75]:


complex_model_2.predict([[1996,7,1000,1200,1200,1,4,1,10]])


# # 6.3 Multiple Linear Regression Model 3 - Selected Features
# ** 13 Features v.s. House Price**

# In[76]:


############################################# Features 3 ##################################################

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features_3=['YearBuilt','OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd']

complex_model_3 = linear_model.LinearRegression()
complex_model_3.fit(train_data_dm[features_3],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_3.intercept_)))
print(('Coefficients: {}'.format(complex_model_3.coef_)))

pred = complex_model_3.predict(test_data_dm[features_3])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rtrcm = float(format(complex_model_3.score(train_data_dm[features_3],train_data_dm['SalePrice']),'.3f'))
rmse=(math.sqrt(msecm))
artrcm = float(format(adjustedR2(complex_model_3.score(train_data_dm[features_3],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_3)),'.3f'))
rtecm = float(format(complex_model_3.score(test_data_dm[features_3],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_3.score(test_data_dm[features_3],test_data['SalePrice']),test_data_dm.shape[0],len(features_3)),'.3f'))
cv = float(format(cross_val_score(complex_model_3,df[features_3],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 3 - 7 Features','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# **Predict house price with model 3**
# * YearBuilt:1999
# * OverallQual: 6
# * TotalBsmtSF:700
# * 1stFlrSF:1200
# * GrLivArea:1050
# * FullBath:1.5
# * TotRmsAbvGrd:3
# * ** Final Predicted Price: $150207.51917045**

# In[77]:


complex_model_3.predict([[1999,6,700,1200,1050,1.5,3]])


# # 6.4 Multiple Linear Regression Model 4 - Selected Features
# ** 12 Features (House Structure)  v.s. House Price**

# In[78]:


############################################# Features 4 ##################################################

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features_4=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

complex_model_4 = linear_model.LinearRegression()
complex_model_4.fit(train_data_dm[features_4],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_4.intercept_)))
print(('Coefficients: {}'.format(complex_model_4.coef_)))

pred = complex_model_4.predict(test_data_dm[features_4])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rmse=(math.sqrt(msecm))
rtrcm = float(format(complex_model_4.score(train_data_dm[features_4],train_data_dm['SalePrice']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_4.score(train_data_dm[features_4],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_4)),'.3f'))
rtecm = float(format(complex_model_4.score(test_data_dm[features_4],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_4.score(test_data_dm[features_4],test_data['SalePrice']),test_data_dm.shape[0],len(features_4)),'.3f'))
cv = float(format(cross_val_score(complex_model_4,df[features_4],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 4 - 12 Features (House Structure) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# **Predict house price with 12 features (house structure) information**

# * 2 basement full bathrooms
# * 1 basement half bathroom
# * 1 full bathroom
# * 2 half bathrooms
# * ...
# * ...
# * **Predicted price: $30326.99351913**

# In[79]:


complex_model_4.predict([[2,1,1,2,1.5,2,0,1,0,0,0,1]])


# # 6.5 Multiple Linear Regression Model 5 - Selected Features
# ** 3 Features (Building Years)  v.s. House Price**

# In[80]:


############################################# Features 5 ##################################################

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features_5=['YearBuilt','YearRemodAdd','GarageYrBlt']

complex_model_5 = linear_model.LinearRegression()
complex_model_5.fit(train_data_dm[features_5],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_5.intercept_)))
print(('Coefficients: {}'.format(complex_model_5.coef_)))

pred = complex_model_5.predict(test_data_dm[features_5])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rmse=(math.sqrt(msecm))
rtrcm = float(format(complex_model_5.score(train_data_dm[features_5],train_data_dm['SalePrice']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_5.score(train_data_dm[features_5],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_5)),'.3f'))
rtecm = float(format(complex_model_5.score(test_data_dm[features_5],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_5.score(test_data_dm[features_5],test_data['SalePrice']),test_data_dm.shape[0],len(features_5)),'.3f'))
cv = float(format(cross_val_score(complex_model_5,df[features_5],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 5 - 3 Features (Building Years) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# **Using Model 5 to predict price for a house which is:
# **
# * built in 1925
# * Renovated in 1950
# * And added garage in 1977

# In[81]:


complex_model_5.predict([[1925,1950,1977]])


# In[82]:


print([features])


# # 6.6 Multiple Linear Regression Model 6 - Selected Features
# ** 9 Features (House Structure + Condition)  v.s. House Price**

# In[83]:


############################################# Features 6 ##################################################

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features_6=['MSSubClass','OverallQual', 'OverallCond','TotalBsmtSF','GrLivArea','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd','MiscVal']

complex_model_6 = linear_model.LinearRegression()
complex_model_6.fit(train_data_dm[features_6],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_6.intercept_)))
print(('Coefficients: {}'.format(complex_model_6.coef_)))

pred = complex_model_6.predict(test_data_dm[features_6])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rmse=(math.sqrt(msecm))
rtrcm = float(format(complex_model_6.score(train_data_dm[features_6],train_data_dm['SalePrice']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_6.score(train_data_dm[features_6],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_6)),'.3f'))
rtecm = float(format(complex_model_6.score(test_data_dm[features_6],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_6.score(test_data_dm[features_6],test_data['SalePrice']),test_data_dm.shape[0],len(features_6)),'.3f'))
cv = float(format(cross_val_score(complex_model_6,df[features_6],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 6 - 9 Features (House Structure + Condition) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# In[84]:


############################################# Features 7 ##################################################

train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features_7=['MSSubClass','OverallQual', 'OverallCond','GrLivArea','BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd']

complex_model_7 = linear_model.LinearRegression()
complex_model_7.fit(train_data_dm[features_7],train_data_dm['SalePrice'])

print(('Intercept: {}'.format(complex_model_7.intercept_)))
print(('Coefficients: {}'.format(complex_model_7.coef_)))

pred = complex_model_7.predict(test_data_dm[features_7])
msecm = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred)),'.3f'))
rmse=(math.sqrt(msesm))
rtrcm = float(format(complex_model_7.score(train_data_dm[features_7],train_data_dm['SalePrice']),'.3f'))
artrcm = float(format(adjustedR2(complex_model_7.score(train_data_dm[features_7],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_7)),'.3f'))
rtecm = float(format(complex_model_7.score(test_data_dm[features_7],test_data_dm['SalePrice']),'.3f'))
artecm = float(format(adjustedR2(complex_model_7.score(test_data_dm[features_7],test_data['SalePrice']),test_data_dm.shape[0],len(features_7)),'.3f'))
cv = float(format(cross_val_score(complex_model_7,df[features_7],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Multiple Regression 7 - 7 Features (House Structure + Condition) v.s. House Price','Selected Features',msecm,rmse,rtrcm,artrcm,rtecm,artecm,cv]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# # 7. KNN Regression

# In[85]:


features_8=['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'MoSold',
 'YrSold',
 'SalePrice']


# In[86]:


knnreg = KNeighborsRegressor(n_neighbors=5)
knnreg.fit(train_data_dm[features_8],train_data_dm['SalePrice'])
pred = knnreg.predict(test_data_dm[features_8])

mseknn1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse1=(math.sqrt(mseknn1))
rtrknn1 = float(format(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),'.3f'))
artrknn1 = float(format(adjustedR2(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_8)),'.3f'))
rteknn1 = float(format(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),'.3f'))
arteknn1 = float(format(adjustedR2(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_8)),'.3f'))
cv1 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))

knnreg = KNeighborsRegressor(n_neighbors=11)
knnreg.fit(train_data_dm[features_8],train_data_dm['SalePrice'])
pred = knnreg.predict(test_data_dm[features_8])

mseknn2 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse2=(math.sqrt(mseknn2))
rtrknn2 = float(format(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),'.3f'))
artrknn2 = float(format(adjustedR2(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_8)),'.3f'))
rteknn2 = float(format(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),'.3f'))
arteknn2 = float(format(adjustedR2(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_8)),'.3f'))
cv2 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))

knnreg = KNeighborsRegressor(n_neighbors=17)
knnreg.fit(train_data_dm[features_8],train_data_dm['SalePrice'])
pred = knnreg.predict(test_data_dm[features_8])

mseknn3 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse3=(math.sqrt(mseknn3))
rtrknn3 = float(format(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),'.3f'))
artrknn3 = float(format(adjustedR2(knnreg.score(train_data_dm[features_8],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_8)),'.3f'))
rteknn3 = float(format(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),'.3f'))
arteknn3 = float(format(adjustedR2(knnreg.score(test_data_dm[features_8],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_8)),'.3f'))
cv3 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['KNN Regression','k=5, all features',mseknn1,rmse1,rtrknn1,artrknn1,rteknn1,arteknn1,cv1]
evaluation.loc[r+1] = ['KNN Regression','k=11, all features',mseknn2,rmse2,rtrknn2,artrknn2,rteknn2,arteknn2,cv2]
evaluation.loc[r+2] = ['KNN Regression','k=17, all features',mseknn3,rmse3,rtrknn3,artrknn3,rteknn3,arteknn3,cv3]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# In[87]:


knnreg = KNeighborsRegressor(n_neighbors=5)
knnreg.fit(train_data_dm[features_2],train_data_dm['SalePrice'])
pred = knnreg.predict(test_data_dm[features_2])

mseknn1 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse1=(math.sqrt(mseknn1))
rtrknn1 = float(format(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))
artrknn1 = float(format(adjustedR2(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))
rteknn1 = float(format(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))
arteknn1 = float(format(adjustedR2(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))
cv1 = float(format(cross_val_score(knnreg,df[features_2],df['SalePrice'],cv=5).mean(),'.3f'))

knnreg = KNeighborsRegressor(n_neighbors=11)
knnreg.fit(train_data_dm[features_2],train_data_dm['SalePrice'])
pred = knnreg.predict(test_data_dm[features_2])

mseknn2 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse2=(math.sqrt(mseknn2))
rtrknn2 = float(format(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))
artrknn2 = float(format(adjustedR2(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))
rteknn2 = float(format(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))
arteknn2 = float(format(adjustedR2(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))
cv2 = float(format(cross_val_score(knnreg,df[features_8],df['SalePrice'],cv=5).mean(),'.3f'))

knnreg = KNeighborsRegressor(n_neighbors=17)
knnreg.fit(train_data_dm[features_2],train_data_dm['SalePrice'])
pred = knnreg.predict(test_data_dm[features_2])

mseknn3 = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rmse3=(math.sqrt(mseknn3))
rtrknn3 = float(format(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),'.3f'))
artrknn3 = float(format(adjustedR2(knnreg.score(train_data_dm[features_2],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features_2)),'.3f'))
rteknn3 = float(format(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),'.3f'))
arteknn3 = float(format(adjustedR2(knnreg.score(test_data_dm[features_2],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features_2)),'.3f'))
cv3 = float(format(cross_val_score(knnreg,df[features_2],df['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['KNN Regression','k=5, selected features',mseknn1,rmse1,rtrknn1,artrknn1,rteknn1,arteknn1,cv1]
evaluation.loc[r+1] = ['KNN Regression','k=11, selected features',mseknn2,rmse2,rtrknn2,artrknn2,rteknn2,arteknn2,cv2]
evaluation.loc[r+2] = ['KNN Regression','k=17, selected features',mseknn3,rmse3,rtrknn3,artrknn3,rteknn3,arteknn3,cv3]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# # 8.
# 

# # Ridge Regressor

# In[88]:


df_dm=df
train_data_dm,test_data_dm = train_test_split(df,train_size = 0.6,random_state=3)

features=['LotArea', 'MasVnrArea', 'BsmtFinSF1', 
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
             'GrLivArea', 'GarageArea', 'WoodDeckSF', 
             'OpenPorchSF', 'PoolArea', 'LotFrontage', 
             'MiscVal', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
             'OverallQual', 'OverallCond', 'YearBuilt']


complex_model_R = linear_model.Ridge(alpha=1)
complex_model_R.fit(train_data_dm[features],train_data_dm['SalePrice'])

pred1 = complex_model_R.predict(test_data_dm[features])
msecm1 = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred1)),'.3f'))
rtrcm1 = float(format(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))
artrcm1 = float(format(adjustedR2(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm1 = float(format(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))
artecm1 = float(format(adjustedR2(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))
cv1 = float(format(cross_val_score(complex_model_R,df_dm[features],df_dm['SalePrice'],cv=5).mean(),'.3f'))

complex_model_R = linear_model.Ridge(alpha=100)
complex_model_R.fit(train_data_dm[features],train_data_dm['SalePrice'])

pred2 = complex_model_R.predict(test_data_dm[features])
msecm2 = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred2)),'.3f'))
rtrcm2 = float(format(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))
artrcm2 = float(format(adjustedR2(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm2 = float(format(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))
artecm2 = float(format(adjustedR2(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))
cv2 = float(format(cross_val_score(complex_model_R,df_dm[features],df_dm['SalePrice'],cv=5).mean(),'.3f'))

complex_model_R = linear_model.Ridge(alpha=1000)
complex_model_R.fit(train_data_dm[features],train_data_dm['SalePrice'])

pred3 = complex_model_R.predict(test_data_dm[features])
msecm3 = float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['SalePrice'],pred3)),'.3f'))
rtrcm3 = float(format(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),'.3f'))
artrcm3 = float(format(adjustedR2(complex_model_R.score(train_data_dm[features],train_data_dm['SalePrice']),train_data_dm.shape[0],len(features)),'.3f'))
rtecm3 = float(format(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),'.3f'))
artecm3 = float(format(adjustedR2(complex_model_R.score(test_data_dm[features],test_data_dm['SalePrice']),test_data_dm.shape[0],len(features)),'.3f'))
cv3 = float(format(cross_val_score(complex_model_R,df_dm[features],df_dm['SalePrice'],cv=5).mean(),'.3f'))

r = evaluation.shape[0]
evaluation.loc[r] = ['Ridge Regression','alpha=1, all features',msecm1,'-',rtrcm1,artrcm1,rtecm1,artecm1,cv1]
evaluation.loc[r+1] = ['Ridge Regression','alpha=100, all features',msecm2,'-',rtrcm2,artrcm2,rtecm2,artecm2,cv2]
evaluation.loc[r+2] = ['Ridge Regression','alpha=1000, all features',msecm3,'-',rtrcm3,artrcm3,rtecm3,artecm3,cv3]
evaluation.sort_values(by = '5-Fold Cross Validation', ascending=False)


# # 8. K-Means Clustering

# In[89]:


from sklearn.cluster import KMeans

temp = df.select_dtypes(include='object')
dumb = pd.get_dummies(temp)
df2 = df.select_dtypes(exclude='object')
df3 = pd.concat([df2, dumb], axis=1, sort=False)

# df3.LotFrontage = df3.LotFrontage.astype(int)
df4 = df3.fillna(0).astype(int)
df_tr = df4

clmns = list(df4.columns.values)

ks = list(range(1, 20))
inertias = []
sse = {}

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(df4)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
    #for plot... please work
    sse[k] = model.inertia_
    



# In[90]:


plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()


# In[91]:


model1 = KMeans(n_clusters=4, random_state=0)
df_tr_std = stats.zscore(df_tr[clmns])


# In[92]:


model1.fit(df_tr_std)
labels = model1.labels_


# In[93]:


df_tr['clusters'] = labels
clmns.extend(['clusters'])


# In[94]:


x1 = df_tr[["YearBuilt","OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","clusters"]]
pd.options.display.max_columns = None

print((x1.groupby(['clusters']).mean()))


# In[95]:


x2 = df_tr[["YearBuilt","OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","clusters"]]
pd.options.display.max_columns = None
print((x1.groupby(['clusters']).std()))

