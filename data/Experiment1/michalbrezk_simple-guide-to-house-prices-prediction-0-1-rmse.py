#!/usr/bin/env python
# coding: utf-8

# # House Prices Prediction - Standard regression problem
# Our task today is to predict prices of houses based on numerous features. Let's try!

# In[1]:


# load libraries we will be using
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,VotingRegressor,StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV, SGDRegressor, ARDRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import xgboost
from sklearn.metrics import mean_squared_error,mean_squared_log_error


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 500


# In[2]:


# load our training dataset
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

df = pd.concat([df_train, df_test], axis=0)
df.head()


# In[3]:


df_test.head()


# In[4]:


df.tail()


# ## Exploratory Data Analysis (EDA)
# Let's play with data now, explore some interesting facts and relations that could be really useful for predictions

# In[5]:


# what is size of our data?
print(('Dataset has {} rows and {} columns'.format(df.shape[0], df.shape[1])))


# In[6]:


# checking data types, number of rows, missing rows
df.info()


# In[7]:


# check summary statistics
df.describe()


# In[8]:


# drop ID as it's not beneficial
#df.drop('Id', axis=1, inplace=True)


# ### Handle missing values
# Based on analysis above we may see some fields have missing values. To make it clear, we will screen just features those has some missing values and how much it does

# In[9]:


null_counts = df.isnull().sum()
full_counts = df.isnull().count()

mc = null_counts[null_counts > 0]
nmc = null_counts[null_counts > 0]/full_counts[null_counts > 0]

ndf = pd.DataFrame([mc, (nmc*100).round(1)], index=['Null Count', 'Null %']).T.sort_values('Null Count', ascending=False)
ndf.style.background_gradient(cmap='PuBu_r', vmin=19, vmax=20, subset=['Null %'])


# **Conclussion?** We've highlighted fields those has missing percentage under 20%. Generally everything under 15% should be dropped, let's check what values we have in fields with high missing values.
# Remember, missing value may not always mean it's missing ;)

# In[10]:


col_mis_high = ['FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC']
col_mis_low  = ['LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrArea', 'MasVnrType', 'Electrical','MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'GarageCars', 'GarageArea', 'SaleType']


# In[11]:


# look on value counts for each field
for c in col_mis_high:
    print(('Unique values for column {}:'.format(c.upper())))
    print((df[c].value_counts()))
    print('----------\n')


# checking descriptions for all these fields, missing value simply means that property does not have that feature. Dropping these fields could be mistake, as it can have impact on price predicted. Let's impute these values with None and check how it is correlated with SalePrice

# In[12]:


# impute missing
for c in col_mis_high:
    df[c].fillna(value='None', inplace=True)


# In[13]:


plt.figure(figsize=(18, 5))
for i,c in enumerate(col_mis_high):
    plt.subplot(1,len(col_mis_high),i+1)
    sns.barplot(x=c, y='SalePrice', data=df)
    plt.yticks([])


# Looks good, it seems some fields has impact on price, now impute fields with missing values less than 20%, will use mode for categorical features and mean for numerical features

# In[14]:


# split data for numerical and categorical, impute and then push back to original dataset
df_missing = df[col_mis_low].copy()
df_missing_cat = df_missing.select_dtypes(include='object')
df_missing_num = df_missing.select_dtypes(include='number')

for c in df_missing_cat.columns:
    df[c] = df_missing_cat[c].fillna(df_missing_cat[c].mode()[0])

for c in df_missing_num.columns:
    df[c] = df_missing_num.fillna(df_missing_num[c].mean())


# In[15]:


# final check, do we have any missing left?
df.isnull().sum()[df.isnull().sum() > 0]


# ### Correlation
# It's time to look how our features correlates to each other as well how they correlate to target variable. Our features should be independed, meaning correlation between features itself should be close to 0

# In[16]:


# take a look on co
corrmat = df.corr().style.background_gradient()


# In[17]:


# Try seaborn heatmap for more condensed view
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), vmin = -0.8, vmax=0.8, annot=False, square=True, cmap='seismic_r')


# **Outcome?** It's obvious that significant part of our features are correlated, thus may not be independed. Also there is pretty strong correlation to target variable (last row), some of them are strongly positive, few have negative correlation

# ### Find outliers
# We wants to check following fields: MiscVal, GrLivArea, EnclosedPorch, BsmtFinSF1

# In[18]:


plt.figure(figsize=(15, 10))
plt.subplot(321)
sns.scatterplot(df['MiscVal'], df['SalePrice'])
plt.subplot(322)
sns.scatterplot(df['GrLivArea'], df['SalePrice'])
plt.subplot(323)
sns.scatterplot(df['EnclosedPorch'], df['SalePrice'])
plt.subplot(324)
sns.scatterplot(df['BsmtFinSF1'], df['SalePrice'])
plt.subplot(325)
sns.scatterplot(df['LotFrontage'], df['SalePrice'])
plt.subplot(326)
sns.scatterplot(df['LotArea'], df['SalePrice'])


# In[19]:


i1 = np.array(df[df['MiscVal']>3000]['Id'])
i2 = np.array(df[df['GrLivArea']>4000]['Id'])
i3 = np.array(df[df['EnclosedPorch']>350]['Id'])
i4 = np.array(df[df['BsmtFinSF1']>3000]['Id'])
i5 = np.array(df[df['LotArea']>70000]['Id'])
i6 = np.array(df[df['LotFrontage']>200]['Id'])

outlier_idx = np.concatenate((i1,i2,i3,i4,i5,i6))
outlier_idx = [x for x in outlier_idx if x not in df_test['Id'].tolist()]
print(('We have found {} outliers!'.format(len(outlier_idx))))
print(outlier_idx)


# In[20]:


df = df[~df['Id'].isin(outlier_idx)]
df_train = df_train[~df_train['Id'].isin(outlier_idx)]


# ### Distributions
# We have a lot of variables, it would take some time to draw pairplot on all of them, let's check just relation to SalePrice
# To do this, we also split our variables to numerical and categorical and will continue checking them and doing analyses separately

# In[21]:


# some of our numerical variables are category IDs (like overall quality etc)

df_cat = df.select_dtypes(exclude='number').copy()
df_int = df[['YearBuilt', 'YearRemodAdd', 'MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']].copy()
df_num = df[['LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']].copy()

print(('Columns in categorical set: {}'.format(df_cat.shape)))
print(('Columns in ID set: {}'.format(df_int.shape)))
print(('Columns in numerical set: {}'.format(df_num.shape)))


# #### Numerical / float values
# First work with float values

# In[22]:


# check correlations to SalePrice ... we need 20 charts
fig = plt.figure(figsize=(15, 15))

for i,c in enumerate(df_num):
    plt.subplot(5, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    sns.regplot(x=df_num[c], y=df['SalePrice'])


# Some of features has strong correlation with low p-value, some of them has strong correlation with high p-value and some of them has low correlation.
# Notice also outliers having in numerous dimensions, might be handled, but we will use RobustScaler later on that is pretty good agains outliers

# In[23]:


# check distributions
fig = plt.figure(figsize=(15, 15))

for i,c in enumerate(df_num):
    plt.subplot(5, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    sns.distplot(df_num[c], kde=False, rug=True)    # kde disabled as there was issue to calculate it for some fields


# In[24]:


# check skewness of our fields
# screen only those that have higher or lower skew than 1, -1
skews = df_num.skew().sort_values()
skew_index = skews[abs(skews) > 0.5].index
skews[abs(skews) > 0.5]


# In[25]:


# take a look on distributions of these fields separately
fig = plt.figure(figsize=(15, 10))

for i,c in enumerate(df_num[skew_index]):
    plt.subplot(4, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    sns.distplot(df_num[c], kde=False, rug=True)    # kde disabled as there was issue to calculate it for some fields


# In[26]:


# transform fields using log1p
df_num[skew_index] = df_num[skew_index].apply(np.log1p)
df['SalePrice'] = df['SalePrice'].apply(np.log1p)
#df['SalePrice'] = RobustScaler().fit_transform(df['SalePrice'].values.reshape(-1,1))


# In[27]:


# take a look on distributions of these fields after transformation
# LotArea is definitely better
fig = plt.figure(figsize=(15, 10))

for i,c in enumerate(df_num[skew_index]):
    plt.subplot(4, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    sns.distplot(df_num[c], kde=False, rug=True)    # kde disabled as there was issue to calculate it for some fields


# In[28]:


# check skewness of our problematic fields
# notice these fields had skewness higher than 0.5
skews = df_num[skew_index].skew().sort_values()
skews


# **Good!** We have fixed skewness at least a bit, it's time to scale our numerical features

# #### Features engineering

# In[29]:


# basement finished percentage
df_num['BsmtFinSF_P'] = (df_num['BsmtFinSF1'] + df_num['BsmtFinSF2'])/(df['TotalBsmtSF'] + 0.01)

# fllor total size & low quality percentage
df_num['TotalFlrSF'] = (df_num['1stFlrSF'] + df_num['2ndFlrSF'])
df_num['FlrSF_P'] = df_num['LowQualFinSF']/(df_num['TotalFlrSF']+0.01)

# porch
df_num['Porch'] = df_num['OpenPorchSF'] + df_num['EnclosedPorch'] + df_num['3SsnPorch'] + df_num['ScreenPorch']

df_num.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','1stFlrSF','2ndFlrSF','LowQualFinSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'], axis=1, inplace=True)


# In[30]:


# bathrooms
df_int['BsmtBath'] = (df_int['BsmtFullBath'] + 0.5*df_int['BsmtHalfBath'])
df_int['Bath'] = (df_int['FullBath'] + 0.5*df_int['HalfBath'])
df_int['BsmtBath_P'] = df_int['Bath']/(df_int['BsmtBath'] + 0.01)

df_int.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'], axis=1, inplace=True)


# In[31]:


rs = RobustScaler()
for c in df_num.columns:
    df_num[c] = rs.fit_transform(df_num[c].values.reshape(-1,1))

#df['SalePrice'] = rs.fit_transform(df['SalePrice'].values.reshape(-1,1))
df_num.head()


# #### PCA
# Some of our features has higher correlation, let's try PCA and check how much fields is needed to explain 0.95 of variance

# In[32]:


# original data shape
df_num.shape


# In[33]:


# Explain 0.95 of variance
pca = PCA(0.95)
df_num_pca = pca.fit_transform(df_num)
df_num_pca.shape


# That's pretty good! we've reduced our data by 5 columns and got 0.95 of variance

# #### Integer / ID values
# We are now done with purely numerical values, let's go to integer values/ids

# In[34]:


df_int.shape


# **Years**? Do we need years? maybe better would be to transform to something different?

# In[35]:


# Do we have some clear correlation? Yes, slightly
sns.regplot(df_int['YearBuilt'], df['SalePrice'],color='red')
sns.regplot(df_int['YearRemodAdd'], df['SalePrice'],color='blue')


# In[36]:


df_int['YearBuilt'] = (2010 - df_int['YearBuilt'])//10*10
df_int['YearRemodAdd'] = (2010 - df_int['YearRemodAdd'])//10*10
df_int['YrSold'] = (2010 - df_int['YrSold'])//10*10


# In[37]:


# yes it's much better!
sns.regplot(df_int['YearBuilt'], df['SalePrice'],color='red')
sns.regplot(df_int['YearRemodAdd'], df['SalePrice'],color='blue')


# In[38]:


df_int['YearBuiltRemod'] = df_int['YearRemodAdd'] + df_int['YearBuilt']
df_int.drop(['YearRemodAdd','YearBuilt'], axis=1, inplace=True)


# In[39]:


# for me it was hard to understand what GarageYrBlt
def garage_year(x):
    y = 1900 + x
    
    if y > 2020:
        y = y-100
        
    return round(y)

help = df_int['GarageYrBlt'].map(garage_year)
df_int['GarageYrBlt'] = (2010 - df_int['GarageYrBlt'])//10*10


# Take a look on impact of month sold, it seems there is almost no correlation.. let's try to reshape month to winter, summer, ...

# In[40]:


sns.regplot(df_int['GarageYrBlt'], df['SalePrice'])


# In[41]:


sns.regplot(df_int['MoSold'], df['SalePrice'])


# In[42]:


def month_sold(x):
    if x == 7:
        return 1
    else:
        return 0


# In[43]:


df_int['SoldSeason'] = df_int['MoSold'].map(month_sold)
df_int.drop(['MoSold'], axis=1, inplace=True)


# In[44]:


sns.regplot(df_int['SoldSeason'], df['SalePrice'])


# Sligtly better? Maybe just a little bit

# In[45]:


for c in df_int.columns:
    df_int[c] = StandardScaler().fit_transform(df_int[c].values.reshape(-1,1))


# In[46]:


df_int.head()


# #### Categorical features
# Now work with categorical values and encode them

# In[47]:


df_cat.head()


# In[48]:


# Checking now unique values we have for each column, we will have to split our fields to candidates for one hot encoding and ordinal values
df_cat.describe(include='all')


# In[49]:


ordinal = ['LandSlope', 'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','KitchenQual', 'HeatingQC','FireplaceQu','GarageFinish','GarageQual','GarageCond','PoolQC','Fence']
onehot = df_cat.columns.tolist()
onehot = [x for x in onehot if x not in ordinal]


# #### One hot encoding

# In[50]:


df_cat_oe = OneHotEncoder(handle_unknown='ignore', sparse=False).fit_transform(df_cat[onehot])
df_cat_oe.shape


# #### Label encoder
# Most of our fields are ordinal, would be mistake to simply encode them, let's do dirty job and encode them manually

# In[51]:


df_cat_le = df_cat[ordinal]
df_cat_le.head()


# In[52]:


df_cat_le.describe(include='all')


# In[53]:


df_cat_le['LandSlope'] = df_cat_le['LandSlope'].map({'Gtl':0, 'Mod':1, 'Sev':2})


# In[54]:


df_cat_le['BsmtQual'] = df_cat_le['BsmtQual'].map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[55]:


df_cat_le['BsmtCond'] = df_cat_le['BsmtCond'].map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[56]:


df_cat_le['BsmtExposure'] = df_cat_le['BsmtExposure'].map({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})


# In[57]:


df_cat_le['BsmtFinType1'] = df_cat_le['BsmtFinType1'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})


# In[58]:


df_cat_le['BsmtFinType2'] = df_cat_le['BsmtFinType2'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})


# In[59]:


df_cat_le['BsmtFinType'] = df_cat_le['BsmtFinType1'] + df_cat_le['BsmtFinType2']
df_cat_le.drop(['BsmtFinType1','BsmtFinType2'], axis=1, inplace=True)


# In[60]:


df_cat_le['KitchenQual'] = df_cat_le['KitchenQual'].map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[61]:


df_cat_le['HeatingQC'] = df_cat_le['HeatingQC'].map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[62]:


df_cat_le['FireplaceQu'] = df_cat_le['FireplaceQu'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[63]:


df_cat_le['GarageFinish'] = df_cat_le['GarageFinish'].map({'NA':0, 'Unf':1, 'RFn':2, 'Fin':3})


# In[64]:


df_cat_le['GarageQual'] = df_cat_le['GarageQual'].map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[65]:


df_cat_le['GarageCond'] = df_cat_le['GarageCond'].map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[66]:


df_cat_le['PoolQC'] = df_cat_le['PoolQC'].map({'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})


# In[67]:


df_cat_le['Fence'] = df_cat_le['Fence'].map({'None':0, 'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4})


# In[68]:


for c in df_cat_le.columns:
    df_cat_le[c] = StandardScaler().fit_transform(df_cat_le[c].values.reshape(-1,1))


# In[69]:


df_cat_le.head()


# In[70]:


print((df_num_pca.shape))
print((df_int.shape))
print((df_cat_oe.shape))
print((df_cat_le.shape))


# In[71]:


df_final = np.concatenate((df_num_pca, df_int, df_cat_oe, df_cat_le.values), axis=1)
df_final.shape


# #### Now exclude our testing dataset we wants to predict

# In[72]:


max_id = df_train.shape[0]

df_final_t = df_final[:max_id]
print(('Training shape: ', df_final_t.shape))
y = df['SalePrice'].iloc[:max_id].values.reshape(-1,1)
print(('Target shape: ', y.shape))

# # prepare feature values
df_predict = df_final[max_id:,]
print(('Features shape: ', df_predict.shape))


# ## Modelling
# We will use Lasso and Ridge

# In[73]:


# split data into train & test size
X_train, X_test, y_train, y_test = train_test_split(df_final_t, y, test_size=0.25, random_state=123)


# In[74]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
selector = SelectFromModel(estimator=LassoCV(n_jobs=-1, random_state=123)).fit(X_train, y_train)
coef = np.abs(selector.estimator_.coef_)
coef = coef>0.0001

X_train = X_train[:, coef]
X_test = X_test[:, coef]
df_predict = df_predict[:,coef]


# In[75]:


# define function to get negative root mean squared error from model
def rmse_cv_train(model):
    kf = KFold(5, random_state=123)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

def rmse_cv_test(model):
    kf = KFold(5, random_state=123)
    rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()


# In[76]:


rg = RidgeCV(alphas=np.linspace(1, 20, 60), cv=KFold(5,random_state=123)).fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(rg)))
print(('Score on test set:', rmse_cv_test(rg)))


# In[77]:


ls = LassoCV(n_alphas=200,random_state=123, cv=KFold(5,random_state=123)).fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(ls)))
print(('Score on test set:', rmse_cv_test(ls)))


# In[78]:


el = ElasticNetCV(n_alphas=200,random_state=123, cv=KFold(5,random_state=123)).fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(el)))
print(('Score on test set:', rmse_cv_test(el)))


# In[79]:


gb = GradientBoostingRegressor().fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(gb)))
print(('Score on test set:', rmse_cv_test(gb)))


# In[80]:


rf = RandomForestRegressor(n_jobs=-1).fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(rf)))
print(('Score on test set:', rmse_cv_test(rf)))


# In[81]:


xg = XGBRegressor(objective='reg:squarederror', n_jobs=-1).fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(xg)))
print(('Score on test set:', rmse_cv_test(xg)))


# In[82]:


sv = SVR().fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(sv)))
print(('Score on test set:', rmse_cv_test(sv)))


# In[83]:


# merge results
y_test_hat_rg = rg.predict(X_test)
y_test_hat_ls = ls.predict(X_test)
y_test_hat_el = el.predict(X_test)
y_test_hat_gb = gb.predict(X_test)
y_test_hat_rf = rf.predict(X_test)
y_test_hat_xg = xg.predict(X_test)

y_test_hat = 0.3*y_test_hat_rg.ravel() + 0.2*y_test_hat_ls.ravel() + 0.2*y_test_hat_el.ravel() + 0.1*y_test_hat_gb.ravel() +  0.1*y_test_hat_rf.ravel() + 0.1*y_test_hat_rg.ravel()

rmse = np.sqrt(mean_squared_error(y_test_hat,  y_test))
print(('RMSE: ', rmse))


# In[84]:


# do the same using voting
vreg = VotingRegressor([
    ('rg', rg), 
    ('ls', ls),
    ('el', el),
    ('gb', gb),
    ('rf', rf),
    ('xg', xg)
])

vr = vreg.fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(vr)))
print(('Score on test set:', rmse_cv_test(vr)))

y_test_hat_vr = vr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_hat_vr,  y_test))
print(('RMSE: ', rmse))


# Those are pretty good scores! Will use manual merge later on after optimizatoin

# #### Hyperparameter tuning
# We got nice scores from different models, let's try to tune them first before merging them together. Note that Ridge, Lasso and ElasticNet are already tuned. I will do several iterrations in each step to get best parameters. Alternatively I could use random grid search to speedup process.

# In[85]:


# support vector machines
# actually we've ended up not using SVM as it looked it's score is not so good as from the others
params = {
    'degree':[1,2,3,4],
    'C':[0.01,0.1,0.5,1,2,5,10],
    'epsilon':[0.001,0.01,0.1,1,5]
}

gssvr = GridSearchCV(SVR(), param_grid=params, n_jobs=-1, scoring='neg_mean_squared_error').fit(X_train, y_train)
print(('SVR best params: ', gssvr.best_params_))
print(('SVR best score: ', gssvr.best_score_))


# In[86]:


# gradient boosting
params = {
    'random_state': [123],
    'max_depth': [1,2],
    'max_features': [8,10],
    'min_samples_leaf': [3,5],
    'min_samples_split': [1,2,3],
    'n_estimators': [1000,1200]
}

gsgb = GridSearchCV(GradientBoostingRegressor(), param_grid=params, n_jobs=-1, scoring='neg_mean_squared_error').fit(X_train, y_train)
print(('Gradient Boosting best params: ', gsgb.best_params_))
print(('Gradient Boosting best score: ', gsgb.best_score_))


# In[87]:


# random forest
params = {
    'random_state': [123],
    'n_jobs': [-1],
    'bootstrap': [True],
    'max_depth': [200, 300],
    'max_features': [30,40],
    'min_samples_leaf': [1,2],
    'min_samples_split': [2,3],
    'n_estimators': [1200]
}

gsrf = GridSearchCV(RandomForestRegressor(), param_grid=params, n_jobs=-1, scoring='neg_mean_squared_error').fit(X_train, y_train)
print(('Random Forest best params: ', gsrf.best_params_))
print(('Random Forest best score: ', gsrf.best_score_))


# In[88]:


# xgb regressor
params = {
    'nthread':[-1],
    'objective':['reg:squarederror'],
    'learning_rate': [0.02,0.03],
    'max_depth': [2,3],
    'min_child_weight': [1,2],
    'subsample': [0.7],
    'colsample_bytree': [0.5,0.6],
    'n_estimators': [1500,2000]
}

gsxb = GridSearchCV(XGBRegressor(), param_grid=params, n_jobs=-1, scoring='neg_mean_squared_error').fit(X_train, y_train)
print(('XGB Regressor best params: ', gsxb.best_params_))
print(('XGB Regressor best score: ', gsxb.best_score_))


# #### Hypertuning outcome
# We have run parameter hypertuning on multiple models, we will now retrain our models to use best parameters and compare score of our prediction once again

# In[89]:


#gbT = GradientBoostingRegressor(max_depth=2, max_features=10, min_samples_leaf=5,min_samples_split=2, n_estimators=500, random_state=123).fit(X_train, y_train)
gbT = gsgb.best_estimator_.fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(gbT)))
print(('Score on test set:', rmse_cv_test(gbT)))


# In[90]:


#rfT = RandomForestRegressor(bootstrap=True, max_depth=200, max_features=30, min_samples_leaf=1, min_samples_split=3, n_estimators=1200, n_jobs=-1, random_state=123).fit(X_train, y_train)
rfT = gsrf.best_estimator_.fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(rfT)))
print(('Score on test set:', rmse_cv_test(rfT)))


# In[91]:


#xgT = XGBRegressor(colsample_bytree=0.5, learning_rate=0.03, max_depth=3, min_child_weight=1, n_estimators=1200, nthread=-1, objective='reg:squarederror', silent=None, subsample=0.7).fit(X_train, y_train)
xgT = gsxb.best_estimator_.fit(X_train, y_train)
print(('Score on train set:', rmse_cv_train(xgT)))
print(('Score on test set:', rmse_cv_test(xgT)))


# In[92]:


# merge results
y_test_hat_rg = rg.predict(X_test)
y_test_hat_ls = ls.predict(X_test)
y_test_hat_el = el.predict(X_test)
y_test_hat_gb = gbT.predict(X_test)
y_test_hat_rf = rfT.predict(X_test)
y_test_hat_xg = xgT.predict(X_test)

y_test_hat = 0.2*y_test_hat_rg.ravel() + 0.2*y_test_hat_ls.ravel() + 0.2*y_test_hat_el.ravel() + 0.1*y_test_hat_gb.ravel() +  0.1*y_test_hat_rf.ravel() + 0.1*y_test_hat_rg.ravel()

rmse = np.sqrt(mean_squared_error(y_test_hat,  y_test))
print(('RMSE: ', rmse))

rmsle = np.sqrt(mean_squared_log_error(y_test_hat,  y_test))
print(('RMSLE: ', rmsle))


# In[93]:


# Predicted vs acutals
sns.scatterplot(x=y_test.ravel(), y=y_test.ravel(), color='blue')
sns.scatterplot(x=y_test.ravel(), y=y_test_hat.ravel(), color='red')


# In[94]:


# Predicted vs acutals
sns.scatterplot(x=y_test.ravel(), y=y_test.ravel(), color='blue')
sns.scatterplot(x=y_test.ravel(), y=y_test_hat_vr.ravel(), color='red')


# ## Predict our test data
# We are done with modelling now, it's time to predict SalePrice for test set

# In[95]:


# predict price
y_pred_rg = rg.predict(df_predict)
y_pred_ls = ls.predict(df_predict)
y_pred_el = el.predict(df_predict)
y_pred_gb = gbT.predict(df_predict)
y_pred_rf = rfT.predict(df_predict)
y_pred_xg = xgT.predict(df_predict)

y_pred = 0.2*y_pred_rg.ravel() + 0.2*y_pred_ls.ravel() + 0.2*y_pred_el.ravel() + 0.2*y_pred_gb.ravel() +  0.1*y_pred_rf.ravel() + 0.1*y_pred_rg.ravel()


# In[96]:


# revert log transformation
y_pred = np.expm1(y_pred)


# In[97]:


# save to csv
df_submission['SalePrice'] = y_pred
df_submission.to_csv('Submission.csv', index = False)
print('Submission saved!')


# In[98]:


df_submission.head()

