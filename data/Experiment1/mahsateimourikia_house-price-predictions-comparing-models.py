#!/usr/bin/env python
# coding: utf-8

# # House Price Predictions

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm, skew

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

import xgboost as xgb
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# ## Exploratory Data Analysis
# Let's explore the dataset for patterns, insights, anomalies, missing values, correlations, and etc.

# In[2]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")


# ### Analysing Features for Correlations and Missing Values

# In[3]:


train.columns


# 
# Let's see which features are numerical and which features are categorical to go on with our analysis:

# In[4]:


numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns


# In[5]:


cat_features = train.select_dtypes(include=[np.object])
cat_features.columns


# #### Correlation Coefficients
# Now let's take a look at the correlation coefficients between numeric features and SalePrice. Here we can use 4 visualization techniques:
# 
# Correlation Heat Map
# Zoomed Heat Map
# Pair Plot
# Scatter Plot

# In[6]:


fig, ax = plt.subplots()
fig.set_size_inches(15,11)
sns.heatmap(train.corr(), cmap="seismic", center=0)


# Let's take a closer look to the most correlated features to the SalePrice:

# In[7]:


k = 10 #number of variables for heatmap
cols = train.corr().nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# From here we can see that there are features that seem to be highly correlated to the SalePrice. Let's take a closer look to them:

# In[8]:


d = train[cols]
sns.pairplot(d, size=2.5)


# ### Removing Outliers

# In[9]:


train.drop(index=train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, axis=0, inplace=True)


# Let's unify the test and train sets to conduct cleaning on the whole data set:

# In[10]:


ntrain = train.shape[0]
ntest = test.shape[0]

ytrain = train['SalePrice']

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(["SalePrice"], axis=1, inplace=True)


# ### Removing the Unuseful Features
# As it can be seen, we have a ID feature which is completely useless for out model. Therefore, we can drop it:

# In[11]:


all_data.drop(['Id'], axis=1, inplace=True)


# 
# We can also drop the highly correlated features as having all of them in our model will not be useful. Highly correlated features can be viewed in the headmap in the previous section. I will remove one of the each highly correlated pair:

# In[12]:


correlated_cols = ['GarageYrBlt', 'GarageArea', 'GarageCond', 'TotRmsAbvGrd', '1stFlrSF', 'YearRemodAdd', 'BsmtUnfSF']

for c in correlated_cols:
    all_data.drop(c, axis = 1, inplace=True)


# In[13]:


all_data.shape


# ## Missing Values
# Let's see how is the situation with the missing value to decide on a strategy about how to resolve them.

# In[14]:


plt.figure(figsize=(20,20))
sns.heatmap(all_data.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# Here we can see that for the Target "SalePrice" we do not have any missing value.
# 
# There are features, i.e., Alley, FireplaceQu, PoolQC, Fence, and MiscFeature, with too many missing values that we hardly can get any useful information from the remaining values. We would probably have to completely drop these features, or to change them to another feature that can be interpreted, such as, a binary feature indicating if we have a value on them or not. To decide a good strategy, more analysis is required.
# 
# There are some other featues, i.e., LotFrontage, MasVnrType, MasVnrArea, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath, GarageType, GarageYrBlt, GarageArea, GarageCars, GarageQual, GarageCond, MSZoning, SaleType, Utilities, Function for which there are some missing values. For these, the proportion of the missing data is small enough to be able to fix them using one of the approaches to replace null values.

# ### Imputing the Missing Values
# Starting with the Alley, FireplaceQu, PoolQC, Fence, and MiscFeature with most missing values, we will try to make a decision about how to manage them:
# 
# - Alley is the "Type of alley access to property" according to the descriptions. And it can get three values:
# ```
#     Grvl    Gravel
#     Pave    Paved
#     NA     No alley access
# ```
#   
# Therefore, the NA value can be interpreted as no alley access to the property. We will replace null values to "None".

# In[15]:


all_data["Alley"] = all_data["Alley"].fillna("None")


# - FireplaceQu indicates the Fireplace quality and can get the following values:
# ```
#     Ex    Excellent - Exceptional Masonry Fireplace
#     Gd    Good - Masonry Fireplace in main level
#     TA    Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#     Fa    Fair - Prefabricated Fireplace in basement
#     Po    Poor - Ben Franklin Stove
#     NA    No Fireplace
# ```     
# We will replace null values to "None".

# In[16]:


all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")


# - PoolQC indicates the Pool quality, and can get the following values:
# ```
#     Ex    Excellent
#     Gd    Good
#     TA    Average/Typical
#     Fa    Fair
#     NA    No Pool
# ```
# We will replace null values to "None".

# In[17]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")


# - Fence indicates the Fence quality, and can get the following values:
# ```
#     GdPrv    Good Privacy
#     MnPrv    Minimum Privacy
#     GdWo    Good Wood
#     MnWw    Minimum Wood/Wire
#     NA    No Fence
# ```
# We will replace null values to "None".

# In[18]:


all_data["Fence"] = all_data["Fence"].fillna("None")


# - MiscFeature is the Miscellaneous feature not covered in other categories:
# ```
#     Elev    Elevator
#     Gar2    2nd Garage (if not described in garage section)
#     Othr    Other
#     Shed    Shed (over 100 SF)
#     TenC    Tennis Court
#     NA    None
# ```
# We will replace null values to "None".

# In[19]:


all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")


# Now let's move to LotFrontage, MasVnrType, MasVnrArea, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath, GarageType, GarageYrBlt, GarageArea, GarageCars, GarageQual, GarageCond, MSZoning, SaleType, Utilities, Function for which there are some missing values.
# 
# - LotFrontage is the linear feet of street connected to property. Here we can replace the missing values by the median or mean of all the LotFrontage values. However a better approach would be to reason to find a better replacement. There is a better chance that the LotFrontage of a house would be close the LotFrontage of the other houses to the same neighborhood. So we can use the median of the LotFrontage of the houses in the neighborhood to replace the missing values

# In[20]:


all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))


# - MasVnrType is the Masonry veneer type, with the following values:
# ```
#  BrkCmn    Brick Common
#  BrkFace    Brick Face
#  CBlock    Cinder Block
#  None    None
#  Stone    Stone
# ```
# - And MasVnrArea is the Masonry veneer area in square feet.
# Looking at the null values they match on both, most probably the null values indicate that there are no Masonry veneers. Therefore, we can replace null MasVnrType with "None" and null MasVnrArea with 0.

# In[21]:


all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")


# In[22]:


all_data["MasVnrArea"]= all_data["MasVnrArea"].fillna(0)


# - BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, These are categorical features about the basement of the house, the Null values most probably indicate that there is no basement in the house. Therefore, we will replace null values with "None" for these features.
# 
# - BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement

# In[23]:


for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
    all_data[col] = all_data[col].fillna("None")


# In[24]:


for col in ('BsmtFinSF1', 'BsmtFinSF2','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)


# - GarageType, GarageFinish, GarageQual, and GarageCond are categorical features related to the Garage of the house. The Null values will most probabily indicate that the house doesn't have a Garage. Therefore, we can replace these values with None.
# 
# - GarageYrBlt is the year garage was built, GarageArea is the Garage Area, GarageCars is the size of garage in car capacity. We replace null values with 0 indicating that there is no Garage.

# In[25]:


for col in ("GarageType", "GarageFinish", "GarageQual"):
    all_data[col] = all_data[col].fillna("None")


# In[26]:


all_data['GarageCars'] = all_data['GarageCars'].fillna(0)


# - MSZoning is the general zoning classification, 'RL' is by far the most common value. So we can fill in missing values with 'RL':

# In[27]:


all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])


# - Utilities For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . We will replace the missing values with "AllPub" which is more common:

# In[28]:


all_data["Utilities"] = all_data["Utilities"].fillna("AllPub")


# - SaleType we can replace the missing value with the most common one:

# In[29]:


all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


# - Last but not the least, Electrical indicates the Electrical system with the following values:
# ```
#  SBrkr    Standard Circuit Breakers & Romex
#  FuseA    Fuse Box over 60 AMP and all Romex wiring (Average)    
#  FuseF    60 AMP Fuse Box and mostly Romex wiring (Fair)
#  FuseP    60 AMP Fuse Box and mostly knob & tube wiring (poor)
#  Mix    Mixed
# ```
# 
# It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
# 
# - Functional is the home functionality, where we should assume typical unless deductions are warranted. Therefore, we can replace the missing value with "Typ" for Typical.

# In[30]:


all_data["Functional"] = all_data["Functional"].fillna("Typ")


# In[31]:


all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])


# ## Features Engineering
# ### Type Corrections
# Let's check and fix the data types that are not correct. We have some numerical data that are actually categorical. These include the following: MSSubClass, OverallCond, YrSold, MoSold. We will transform them to categorical features:

# In[32]:


all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].apply(str)
all_data['YrSold'] = all_data['YrSold'].apply(str)
all_data['MoSold'] = all_data['MoSold'].apply(str)


# ### Label Encoding
# We use label encoding on categorical features, to help normalize labels such that they contain only values between 0 and n_classes-1. And to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels:

# In[33]:


cols= list(all_data.select_dtypes(include=[np.object]).columns)

from sklearn.preprocessing import LabelEncoder


for c in cols:
    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(list(all_data[c].values))
    all_data[c] = lbl_encoder.transform(list(all_data[c].values))


# ### Feature Skewness
# When we talk about normality what we mean is that the data should look like a normal distribution. This is important because several statistic tests rely on this (e.g. t-statistics).
# 
# If the dataset is skewed, then the Machine Learning model wouldn’t be able to do a good job on predictions. To resolve the issue of the skewed features we can apply the a log transform of the same data, or to use the Box-Cox Transformation. Let's see how skewed are our numerical features:

# In[34]:


numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = all_data[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_features_df = pd.DataFrame(skewed_features, columns={'Skew'})
skewed_features_df.head(10)


# In[35]:


skewed_features_df.tail(10)


# In[36]:


skewed_features_df = skewed_features_df[abs(skewed_features_df) > 0.75]

from scipy.special import boxcox1p
lam = 0.15
cols = skewed_features_df.index

for c in cols:
    all_data[c] = boxcox1p(all_data[c], lam)


# ### Convert Categorical Features
# This is to convert categorical variable into dummy/indicator variables

# In[37]:


all_data = pd.get_dummies(all_data, drop_first=True)


# ### Analysing the Target
# Let's take a look at the distribution of the SalePrice which is the value that we are going to predict. One of the points that we would like to see here is to discover anomalies in our train data which would affect our final model. One example is to see if we have a min 0, or a max value which is too high.

# In[38]:


ytrain.describe()


# In[39]:


colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.set_style('whitegrid')
sns.set_palette(sns.xkcd_palette(colors))
plt.figure(figsize=(15,8))
sns.distplot(ytrain, bins=100, fit=norm)

fig = plt.figure()
res = stats.probplot(ytrain, plot=plt)


# We can see that the SalePrice is skewed to the right, and has some pickiness. As can be seen, the house prices are mainly between 80,000 and 300,000. and there are rare prices more than 400,000.
# 
# Let's see the Skewness and Kurtosis measurs of the distribution, however, they wouldn't give us more information rather than the ones we could observe on the histogram.

# In[40]:


print(("Skewness: {0:1.3f}".format(ytrain.skew())))
print(("Kurtosis: {0:1.3f}".format(ytrain.kurt())))


# We should also resolve the skewness in the results:[](http://)

# In[41]:


ytrain = boxcox1p(ytrain, lam)


# In[42]:


sns.distplot(ytrain, bins=100)

fig = plt.figure()
res = stats.probplot(ytrain, plot=plt)


# In[43]:


xtrain = all_data[:ntrain]
xtest = all_data[ntrain:]


# ### Check for Homoscedasticity
# Homoscedasticity refers to the 'assumption that dependent variable(s) exhibit equal levels of variance across the range of predictor variable(s)' (Hair et al., 2013). Homoscedasticity is desirable because we want the error term to be the same across all values of the independent variables.
# 
# The best approach to test homoscedasticity for two metric variables is graphically. Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph, large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).
# 
# As we have fixed the skewness in our features this issue should be fixed too. For example we can we see the scatter plot of the GrLivArea and the SalePrice that previously we saw that was cone-shaped (check the pairplot at the begining of the analysis).

# In[44]:


sns.jointplot(xtrain['GrLivArea'], ytrain)


# ## Modelling
# Now we are going to try different regression approaches and evaluate them to choose the best model to apply.
# 
# ### Defining the Cross-Validation Strategy
# First we define a 5-folds cross-validation strategy to evaluate our models:

# In[45]:


k_folds = 5

def cval_score(model):
    kf = KFold(k_folds, shuffle = True, random_state = 101).get_n_splits(train.values)
    score = cross_val_score(model, xtrain.values, ytrain, cv = kf)
    return(score)


# ### Lasso Regression
# As we have many features, I would like to try lasso which also performs feature selection and regularization that helps me to remove the unuseful features to have a less complex and better performing model.
# 
# This model may be very sensitive to outliers. To resolve this issue we can use the sklearn's Robustscaler() method on the pipeline.

# In[46]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.005, random_state=101))
score = cval_score(lasso)
print(("Accuracy: {:0.2f} (+/- {:0.2f})".format(score.mean(), score.std()*2)))


# In[47]:


lasso.fit(xtrain, ytrain)
lasso_predictions = lasso.predict(xtest)


# ### Elastic Ridge Regression
# In statistics and, in particular, in the fitting of linear or logistic regression models, the elastic net is a regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.

# In[48]:


elastic_net_regression = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=0.2, 
                                                              random_state=101))
score = cval_score(elastic_net_regression)
print(("Accuracy: {:0.2f} (+/- {:0.2f})".format(score.mean(), score.std()*2)))


# In[49]:


elastic_net_regression.fit(xtrain, ytrain)
elastic_net_predictions = elastic_net_regression.predict(xtest)


# Kernel Ridge Regression
# Kernel ridge regression (KRR) combines Ridge Regression (linear least squares with l2-norm regularization) with the kernel trick. It thus learns a linear function in the space induced by the respective kernel and the data. For non-linear kernels, this corresponds to a non-linear function in the original space.
# 
# The form of the model learned by KernelRidge is identical to support vector regression (SVR). However, different loss functions are used: KRR uses squared error loss while support vector regression uses $\epsilon$-insensitive loss, both combined with l2 regularization. In contrast to SVR, fitting KernelRidge can be done in closed-form and is typically faster for medium-sized datasets. On the other hand, the learned model is non-sparse and thus slower than SVR, which learns a sparse model for $\epsilon \gt 0$, at prediction-time.

# In[50]:


kernel_ridge_regression = KernelRidge(alpha=1, kernel='polynomial', degree=2)
score = cval_score(kernel_ridge_regression)
print(("Accuracy: {:0.2f} (+/- {:0.2f})".format(score.mean(), score.std()*2)))


# In[51]:


kernel_ridge_regression.fit(xtrain, ytrain)
kernel_ridge_predictions = kernel_ridge_regression.predict(xtest)


# ### Gradient Boosting Regression
# GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function. Using the Huber loss it is robust to outliers

# In[52]:


gradient_boosting_regression = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = cval_score(gradient_boosting_regression)
print(("Accuracy: {:0.2f} (+/- {:0.2f})".format(score.mean(), score.std()*2)))


# In[53]:


gradient_boosting_regression.fit(xtrain, ytrain)
gradient_boosting_predictions = gradient_boosting_regression.predict(xtest)


# ### XGBoost
# XGBoost stands for “Extreme Gradient Boosting”, where the term “Gradient Boosting” originates from the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman. XGBoost provides a parallel tree boosting (also known as GBDT, GBM).

# In[54]:


xgboost = xgb.XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=2000, 
                             colsample_bytree=0.5, gamma=0.5,  
                             min_child_weight=2,
                             reg_alpha=0.5, reg_lambda=1,
                             subsample=0.5, silent=True,
                             random_state =101, nthread = -1)
score = cval_score(xgboost)
print(("Accuracy: {:0.2f} (+/- {:0.2f})".format(score.mean(), score.std()*2)))


# In[55]:


xgboost.fit(xtrain, ytrain)
xgboost_predictions = xgboost.predict(xtest)


# ### LightGBM
# LightGBM is a gradient boosting framework that uses tree based learning algorithms.

# In[56]:


light_gbm = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves = 5, max_depth = 3, 
                              min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11,
                              learning_rate = 0.05, n_estimators = 1000, 
                              objective='regression', bagging_fraction = 0.8, bagging_freq = 5,
                              feature_fraction=0.5, feature_fraction_seed=9, 
                              lambda_l1=0.1, lambda_l2=0.5, min_gain_to_split=0.2,
                              random_state=101, silent=True, 
                              )

score = cval_score(light_gbm)
print(("Accuracy: {:0.2f} (+/- {:0.2f})".format(score.mean(), score.std()*2)))


# In[57]:


light_gbm.fit(xtrain, ytrain)
light_gbm_predictions = light_gbm.predict(xtest)


# ### Stacking
# Stacking, also called Super Learning or Stacked Regression, is a class of algorithms that involves training a second-level “metalearner” to find the optimal combination of the base learners. Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together.
# 
# Stacking is important because it has been found to improve performance in various machine learning problems.
# 
# Here I simply use a weighted average of the trained models as a ensemble method. I use weights to give more importance to predictions of models with a higher accuracy.

# In[58]:


average_predictions = (1*lasso_predictions+ 0.5*elastic_net_predictions+ 0.5*kernel_ridge_predictions+2*gradient_boosting_predictions+ 3*xgboost_predictions+ 3*light_gbm_predictions)/10


# In[59]:


from scipy.special import inv_boxcox1p
final_predictions = inv_boxcox1p(average_predictions, lam)


# ## The Submission
# Now we create the submission file including the test id and predictions:

# In[60]:


submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['SalePrice'] = final_predictions
submission.to_csv('submission2.csv',index=False)


# In[ ]:




