#!/usr/bin/env python
# coding: utf-8

#   # <center>  House Prices </center>
#   

# ### Group Number: 8
# ### Group members:
# - Abdulrahman ALQannas
# - Doaa Alsenani
# - Ghida Qahtan
# - Moayad Magadmi
# ----

# ## Introduction
# 

# These datasets include information about residential homes that were sold from 2006 to 2010 in Ames, Iowa. Our purpose will be to predict the final price of each home.

# ### These datasets include 79 explanatory variables:
# 
# *   Train data have SalePrice (dependent variable) and other predictor variables.
# *   Test data include the same  variables  that in train data, but  without SalePrice (dependent variable) because this data will be submitted to kaggle.
# 
#  

# ## Importing packages

# In[61]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
#sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing, svm
from sklearn.feature_selection import SelectFromModel
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))


# In[62]:


# display all
from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# ## Loading the House Price Dataset

# In[63]:


train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv', index_col='Id')

# combine train and test data 
df_all = pd.concat([train_data.drop('SalePrice', axis=1), test_data], sort=True)  #df without the target





# ## Exploring the Data

# In[64]:


df_all.head()


# In[65]:


df_all.shape


# In[66]:


df_all.info()


# - ### Data Types

# In[67]:


numeric_features = df_all.select_dtypes(include=['int64','float64'])
categorical_features = df_all.select_dtypes(include=['object'])


# In[68]:


print(('Numeric Features:',len(list(numeric_features.columns))))
print(('Categorical Features:',len(list(categorical_features.columns))))


# -  #### This dataset includes 36 numeric features: 
# 
# |Classification|Dataset|Description|
# |-------|---|---|
# |Years/Months|df_all|These numeric features represent time of built, sold, and the age of the property.| 
# |Area|df_all|These features show the square footage| 
# |Amount of Rooms and Amenities|df_all|These features represent the number of rooms, bathrooms,kitchens...ect.| 
# |Condition and Quality|df_all|Theses features show the condition and quality of land that are determined by surveyors| 
# 

# ### Check Missing Values

# In[69]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 6))
sns.heatmap(train_data.isnull(), yticklabels=False, ax = ax[0], cbar=False, cmap='viridis')
ax[0].set_title('Trian data')

sns.heatmap(test_data.isnull(), yticklabels=False, ax = ax[1], cbar=False, cmap='viridis')
ax[1].set_title('Test data');


# ##### Out of 81 features, 19 features have missing values.

# In[70]:


missing = df_all.isnull().sum().sort_values(ascending=False)
percentage=(df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing,percentage],axis=1,keys=['Missing','%']) 
missing_data[missing_data != 0].dropna()


# ### Cleaning Data 

#   - ####   Converting NaN Numeric Values with KNNImputer

# If you have Sklearn 0.22.2 you can use this code in this cell

# In[71]:


# numeric_features = df_all.loc[:, df_all.dtypes != np.object]
# imputer = KNNImputer(n_neighbors=60)
# df_all.loc[:, df_all.dtypes != np.object] = imputer.fit_transform(numeric_features)


# If you don't have sklearn 0.22.2 comment the cell above and use this cell

# In[72]:


imp = SimpleImputer(missing_values=np.nan, strategy='median')
df_all.loc[:, df_all.dtypes != np.object] = imp.fit_transform(numeric_features)
df_all.head()


# - ####   Filling Missing Data

# In[73]:


edit_values = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType',
              'GarageQual','PoolQC','Fence','MiscFeature','MasVnrType', 'GarageCond', 'GarageFinish']

for col in edit_values:
    df_all[col].fillna('NA',inplace=True)


# In[74]:


df_all.Exterior1st.fillna(value='VinylSd', inplace=True)

df_all.Exterior2nd.fillna(value='VinylSd', inplace=True)

df_all.KitchenQual.fillna(value='TA', inplace=True)

df_all.SaleType.fillna(value='WD', inplace=True)

df_all.Utilities.fillna(value='AllPub', inplace=True)

df_all.Electrical.fillna(value='SBrkr', inplace=True)

df_all.Functional.fillna(value='Typ', inplace=True)

df_all.MSZoning.fillna(value='RL', inplace=True)


# In[75]:


# Missung values 
print(('Missing values:' ,df_all.isnull().sum().sum()))


# - ### Change the Data Type

# In[76]:


df_all['GarageYrBlt']= df_all['GarageYrBlt'].astype(int)


# ### Distribution of SalePrice

# Let's take a look at the most important variable <b>SalePrice</b> It seems there is a long tail to the right which means high sale prices, which will make the mean to be much higher than the median.<br><br>

# In[77]:


# adding the target to our df
df_all = pd.concat([df_all, train_data['SalePrice']], axis=1)
normal_sp = df_all['SalePrice'].dropna().map(lambda i: np.log(i) if i > 0 else 0)
print((df_all['SalePrice'].skew()))
print((normal_sp.skew()))

fig, ax = plt.subplots(ncols=2, figsize=(12,6))
df_all.hist('SalePrice', ax=ax[0])
normal_sp.hist(ax=ax[1])
plt.show


# - ### SalePrice Correlation Matrix 

# Now let's take a look at the most important variables, which will have strong linear releationship with <b>SalePrice</b> variable.<br><br>

# In[78]:


fig, ax = plt.subplots(figsize=(14, 8))
k = 10 #number of variables for heatmap
corrmat = df_all.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_all[cols].dropna().values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                 yticklabels=cols.values, xticklabels=cols.values, ax=ax, cmap="YlGnBu")
ax.set_ylim(0 ,10)
plt.show()


# In[79]:


# correlation with the target
corr_matrix = df_all.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)


# ### Outliars

# In[80]:


fig, axes = plt.subplots(ncols=4, nrows=4, 
                         figsize=(5 * 5, 5 * 5), sharey=True)
axes = np.ravel(axes)
cols = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual',
        'BsmtCond','GarageQual','GarageCond', 'MSSubClass','MSZoning',
        'Neighborhood','BldgType','HouseStyle','Heating','Electrical','SaleType']
for i, c in zip(np.arange(len(axes)), cols):
    ax = sns.boxplot(x=c, y='SalePrice', data=df_all, ax=axes[i], palette="Set2")
    ax.set_title(c)
    ax.set_xlabel("")


#  As we see in the correlation matrix, the features that related to quality and the size affect the sale price, which might affect our results. In addition, OverallQual feature has a significant impact on sale price more than other features.
# 

# In[81]:


Q1 = df_all.quantile(0.25)
Q3 = df_all.quantile(0.75)
IQR = Q3 - Q1
outliars = (df_all < (Q1 - 5 * IQR)) | (df_all > (Q3 + 5 * IQR))
#removing bad columns and outliars
no_outliars_df = df_all.drop(['EnclosedPorch', 'KitchenAbvGr'], axis=True)
rm_rows = ['LotArea', 'MasVnrArea', 'PoolArea', 'OpenPorchSF', 'LotFrontage', 'TotalBsmtSF','1stFlrSF',
           'GrLivArea', 'BsmtFinSF1', 'WoodDeckSF']
df_all.drop(['EnclosedPorch', 'KitchenAbvGr'], axis=True, inplace=True)
for row in rm_rows:
    no_outliars_df.drop(no_outliars_df[row][outliars[row]].index, inplace=True)


# ## Categoriesing features

# In[82]:


object_features =df_all.loc[:, df_all.dtypes == np.object]   #object features
df_all= pd.get_dummies(df_all, columns=object_features.columns.values, drop_first=True)
no_outliars_df = pd.get_dummies(no_outliars_df, columns=object_features.columns.values, drop_first=True)


# 
# ### Separating the Dataset After Cleaning
# 
# 
# 

# In[83]:


# Traing Data with Outliers
newtraining=df_all.loc[  : 1460]
# Testing Data with Outliers
newtesting=df_all.loc[1461 : ].drop('SalePrice', axis=1)
# newtraining['SalePrice'] = np.log(newtraining['SalePrice'])
# lab_enc = preprocessing.LabelEncoder()
# newtraining['SalePrice'] = la


# In[84]:


# Traing Data without Outliers
no_outliars_training = no_outliars_df.loc[  : 1460]
# Testing Data without Outliers
no_outliars_test = no_outliars_df.loc[1461 : ].drop('SalePrice', axis=1)


# ### Modeling

# - ##### Splitting and Standardizing Train Data to Obtain Test Scores Without Removing the Outliers

# In[85]:


y = newtraining['SalePrice']
X = newtraining.drop('SalePrice', axis=1)


# In[86]:


ss = StandardScaler()
Xs =ss.fit_transform(X)


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(
    Xs, y, test_size=.3, random_state=1)


# - ##### Splitting and Standardizing Train Data to Obtain Test Scores With Removing the Outliers
# 

# In[88]:


yo = no_outliars_training['SalePrice']
Xo = no_outliars_training.drop('SalePrice', axis=1)


# In[89]:


sso = StandardScaler()
Xso =sso.fit_transform(Xo)


# In[90]:


Xo_train, Xo_test, yo_train, yo_test = train_test_split(
    Xso, yo, test_size=30, random_state=1)


#    ### 1 - Build Linear Regression Model

#   - #### Build the Model With Outliers

# In[91]:


lr = LinearRegression()
model= lr.fit(X_train, y_train)
print(('Train Score:',model.score(X_train, y_train)))
print(('Test Score :',model.score(X_test, y_test)))


# In[92]:


lr_predictions = model.predict(newtesting) 


# - #### Build the Model Without Outliers

# In[93]:


lro = LinearRegression()
model_o= lro.fit(Xo_train, yo_train)
print(('Train Score:', model_o.score(Xo_train, yo_train)))
print(('Test Score :',model_o.score(Xo_test, yo_test)))


# In[94]:


lro_predictions = model_o.predict(Xso) 


#    ### 2 - Build Lasso Model

# - #### Build the Model With Outliers

# In[95]:


lasso = Lasso(alpha=.0002)
lasso.fit(X_train, y_train)
print(('Train Score:',lasso.score(X_train, y_train)))
print(('Test Score: ', lasso.score(X_test, y_test)))


# In[96]:


lasso_predictions = lasso.predict(newtesting)


# In[97]:


# sqrt(mean_squared_error(submission['SalePrice'],lasso_predictions))


# - #### Build the Model Without Outliers

# In[98]:


lasso_o = Lasso(alpha=.2)
lasso_o.fit(Xo_train, yo_train)
print(('Train Score:',lasso_o.score(Xo_train, yo_train)))
print(('Test Score:',lasso_o.score(Xo_test, yo_test)))


#  ### 3 - Build LassoCV Model

# - #### Build the Model With Outliers

# In[99]:


lasso_cv = LassoCV(alphas=np.logspace(-4, 4, 1), cv=5,random_state=1)
lasso_cv.fit(X_train, y_train)
print(('Train Score :',lasso_cv.score(X_train, y_train)))
print(('Test Score:',lasso_cv.score(X_test, y_test)))


# In[100]:


lasso_cv_predictions = lasso_cv.predict(newtesting) 


# - #### Build the Model Without Outliers

# In[101]:


lasso_cv_o = LassoCV(cv=10,random_state=1)
lasso_cv_o.fit(Xo_train, yo_train)
print(('Train Score:',lasso_cv_o.score(Xo_train, yo_train)))
print(('Test Score:',lasso_cv_o.score(Xo_test, yo_test)))


# 
#  ### 4 - Build Ridge Model

# - #### Build the Model With Outliers

# In[102]:


ridge = Ridge(alpha=1) 
ridge.fit(X_train, y_train)
print(('Train Score:',ridge.score(X_train, y_train)))
print(('Test Score:',ridge.score(X_test, y_test)))


# In[103]:


ridge_predictions = ridge.predict(newtesting) 


# - #### Build the Model Without Outliers
# 

# In[104]:


ridge_o = Ridge(alpha=.01) 
ridge_o.fit(Xo_train, yo_train)
print(('Train Score:',ridge_o.score(Xo_train, yo_train)))
print(('Test Score:',ridge_o.score(Xo_test, yo_test)))


#  ### 5 - Build RidgeCV Model

# - #### Build the Model With Outliers

# In[105]:


ridgecv = RidgeCV(alphas=np.logspace(-4, 4, 1))
ridgecv.fit(X_train, y_train)
print(('Train Score:',ridgecv.score(X_train, y_train)))
print(('Test Score:',ridgecv.score(X_test, y_test)))


# In[106]:


ridgeCV_predictions = ridgecv.predict(newtesting) 


# - #### Build the Model Without Outliers

# In[107]:


ridgecv_o = RidgeCV(alphas=np.logspace(-4, 4, 1)) 
ridgecv_o.fit(Xo_train, yo_train)
print(('Train Score:',ridgecv_o.score(Xo_train, yo_train)))
print(('Test Score:',ridgecv_o.score(Xo_test, yo_test)))


# 
#  ### 6 - Build ElasticNet Model 
# 

# - #### Build the Model With Outliers

# In[108]:


elastic=ElasticNet(.00001)
elastic = elastic.fit(X_train, y_train)
print(('Train Score:',elastic.score(X_train, y_train)))
print(('Test Score:',elastic.score(X_test, y_test)))


# In[109]:


elastic_predictions = elastic.predict(newtesting) 


# - #### Build the Model Without Outliers
# 

# In[110]:


elastic_o=ElasticNet(.00001)
elastic_o = elastic_o.fit(Xo_train, yo_train)
print(('Train Score:',elastic_o.score(Xo_train, yo_train)))
print(('Test Score:',elastic_o.score(Xo_test, yo_test)))


# 
#  ### 7 - Build ElasticNetCV Model 

# - #### Build the Model Without Outliers

# In[111]:


elastic_cv=ElasticNetCV(alphas=np.logspace(-10, 6, 1))
elastic_cv = elastic_cv.fit(X_train, y_train)
print(('Train Score:',elastic_cv.score(X_train, y_train)))
print(('Test score:',elastic_cv.score(X_test, y_test)))


# - #### Build the Model Without Outliers

# In[112]:


elastic_cv_o=ElasticNetCV(alphas=np.logspace(-4, 4, 1))
elastic_cv_o = elastic_cv_o.fit(Xo_train, yo_train)
print(('Train Score :',elastic_cv_o.score(Xo_train, yo_train)))
print(('Test score  :',elastic_cv_o.score(Xo_test, yo_test)))


# 
#  ### 8 - Build Decision Tree Model 

# - #### Build the Model With Outliers
# 

# In[113]:


tree = DecisionTreeClassifier(max_depth = 28)
tree.fit(X, y)
print(('Score : ',tree.score(X, y)))


# In[114]:


tree_predictions = tree.predict(newtesting)


# In[115]:


# sqrt(mean_squared_error(submission['SalePrice'], tree_predictions))


# - #### Build the Model Without Outliers

# In[116]:


tree_o = DecisionTreeClassifier(max_depth = 29)
tree_o.fit(Xo, yo)
print(('Score : ',tree_o.score(Xo, yo)))


# 
#  ### 9 - Build Random Forest Model 

# - #### Build the Model With Outliers

# In[117]:


randomF = RandomForestRegressor(max_depth=50)
randomF.fit(X, y)
print(('Train score :',randomF.score(X, y)))


# In[118]:


# from sklearn.model_selection import KFold
# cv=KFold(n_splits=5, shuffle=True, random_state=1)


# In[119]:


# cross_val_score(randomF, X, y, cv=cv)


# In[120]:


# cross_val_score(randomF, X, y, cv=cv).mean()


# In[121]:


randomF_predictions = randomF.predict(newtesting) 


# In[122]:


# sqrt(mean_squared_error(submission_tree['SalePrice'], randomF_predictions))


# GridSearch

# In[123]:


# param_grid = {
#     'n_estimators': [i for i in range(50,1000,10)],
#     'max_features': [86],
#     'max_depth' :[i for i in range(1,100,1)]
# }


# In[124]:


# rand = RandomForestRegressor(n_jobs=-1)

# gs = GridSearchCV(rand, 
#                   param_grid, 
#                   cv=5)


# In[125]:


# gs.fit(X, y)


# In[126]:


# gs.best_params_


# In[127]:


# gs.best_score_


# - #### Build the Model Without Outliers

# In[128]:


randomF_o = RandomForestRegressor(max_depth = 50)
randomF_o.fit(Xo, yo)
print(('train score : ',randomF_o.score(Xo, yo)))


#  ### 10 - Build KNeighborsRegressor Model 

# - #### Build the Model With Outliers

# In[129]:


neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(X_train, y_train)
print(('Train score : ',neigh.score(X_train, y_train)))
print(('Test score  : ',neigh.score(X_test, y_test)))


# In[130]:


neigh_predictions = neigh.predict(newtesting)


# In[131]:


# sqrt(mean_squared_error(submission_tree['SalePrice'], neigh_predictions))


# - #### Build the Model Without Outliers

# In[132]:


neigh_o = KNeighborsRegressor(n_neighbors=3)
neigh_o.fit(Xo_train, yo_train)
print(('Train score : ',neigh_o.score(Xo_train, yo_train)))
print(('Test score  : ',neigh_o.score(Xo_test, yo_test)))


#  ### 11 - Build SVM  Model 
# 

# In[133]:


svm_l = svm.SVC(kernel='linear')
svm_l.fit(X, y)
svm_l.score(X, y)


# In[134]:


cv=KFold(n_splits=5, shuffle=True, random_state=1)
cross_val_score(svm_l, X, y, cv=cv, n_jobs=-1).mean()


# In[135]:


svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X, y)
svm_rbf.score(X, y)
cross_val_score(svm_rbf, X, y, cv=5, n_jobs=-1).mean()


# In[136]:


svm_p = svm.SVC(kernel='poly')
svm_p.fit(X, y)
svm_p.score(X, y)
cross_val_score(svm_p, X, y, cv=5, n_jobs=-1).mean()


# In[137]:


svm_rbf = svm.SVC(kernel='rbf', gamma=0.001)
cross_val_score(svm_rbf, X, y, cv=5).mean()


# ###  Results

# 
# - #### Creating a list to Store Score Values for Each Model
#      -  This list will display the train and test scores without outliers

# In[138]:


list_of_Scores = list()


# In[139]:


# LinearRegression
results = {'Model':'LinearRegression',
           'Train Score': model.score(X_train, y_train),
           'Test Score':model.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)
# Lasso
results = {'Model':'Lasso',
           'Train Score':lasso.score(X_train, y_train),
           'Test Score': lasso.score(X_test, y_test),
           'Kaggle Score':0.59683}
list_of_Scores.append(results)
# LassoCV
results = {'Model':'LassoCV',
           'Train Score': lasso_cv.score(X_train, y_train),
           'Test Score':lasso_cv.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# Ridg
results = {'Model':'Ridg',
           'Train Score': ridge.score(X_train, y_train),
           'Test Score':ridge.score(X_test, y_test),
           'Kaggle Score':0.36706}
list_of_Scores.append(results)

# RidgCV
results = {'Model':'RidgCV',
           'Train Score': ridgecv.score(X_train, y_train),
           'Test Score':ridgecv.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# ElasticNet
results = {'Model':'ElasticNet',
           'Train Score': elastic.score(X_train, y_train),
           'Test Score':elastic.score(X_test, y_test),
           'Kaggle Score':6.63994}
list_of_Scores.append(results)

# ElasticNetCV
results = {'Model':'ElasticNetCV',
           'Train Score':elastic_cv.score(X_train, y_train),
           'Test Score':elastic_cv.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# DecisionTreeRegressor
results = {'Model':'DecisionTreeRegressor',
           'Train Score':tree.score(X, y),
           'Test Score':None,
           'Kaggle Score':0.25525}
list_of_Scores.append(results)

# RandomForest
results = {'Model':'RandomForest',
           'Train Score':randomF.score(X, y),
           'Test Score':None,
           'Kaggle Score':0.14824}
list_of_Scores.append(results) 

# KNeighborsRegressor
results = {'Model':'KNeighborsRegressor',
           'Train Score': neigh.score(X_train, y_train),
           'Test Score':neigh.score(X_test, y_test),
           'Kaggle Score':None}
list_of_Scores.append(results)

# SVM
results = {'Model':'SVM',
           'Train Score': svm_l.score(X, y),
           'Test Score':None,
           'Kaggle Score':None}
list_of_Scores.append(results)


# In[140]:


df_results = pd.DataFrame(list_of_Scores)


# - #### This table provides all the scores that we got from each model.
# 

# In[141]:


df_results


# - #### Creating a list to Store Score Values for Each Model
#      -  This list will display the train and test scores without outliers
# 

# In[142]:


list_of_Scores_o = list()


# In[143]:


# LinearRegression
results_o = {'Model':'LinearRegression',
           'Train Score': lro.score(Xo_train, yo_train),
           'Test Score':lro.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)


# Lasso
results_o = {'Model':'Lasso',
           'Train Score': lasso_o.score(Xo_train, yo_train),
           'Test Score':lasso_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# LassoCv
results_o = {'Model':'LassoCv',
           'Train Score': lasso_cv_o.score(Xo_train, yo_train),
           'Test Score':lasso_cv_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# Ridg
results_o = {'Model':'Ridg',
           'Train Score':ridge_o.score(Xo_train, yo_train),
           'Test Score':ridge_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# RidgCV
results_o = {'Model':'RidgCV',
           'Train Score':ridgecv_o.score(Xo_train, yo_train),
           'Test Score':ridgecv_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# ElasticNet
results_o = {'Model':'ElasticNet',
           'Train Score':elastic_o.score(Xo_train, yo_train),
           'Test Score':elastic_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# ElasticNetCV
results_o = {'Model':'ElasticNetCV',
           'Train Score':elastic_cv_o.score(Xo_train, yo_train),
           'Test Score':elastic_cv_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)

# DecisionTreeRegressor
results_o = {'Model':'DecisionTreeRegressor',
           'Train Score':tree_o.score(Xo, yo),
           'Test Score':None}
list_of_Scores_o.append(results_o)

# RandomForest
results_o = {'Model':'RandomForest',
           'Train Score':randomF_o.score(Xo, yo),
           'Test Score':None}
list_of_Scores_o.append(results_o)

# KNeighborsRegressor
results_o = {'Model':'KNeighborsRegressor',
           'Train Score':neigh_o.score(Xo_train, yo_train),
           'Test Score':neigh_o.score(Xo_test, yo_test)}
list_of_Scores_o.append(results_o)


# In[144]:


df_results_o = pd.DataFrame(list_of_Scores_o)


# * #### This table provides all the scores that we got from each model.
# 

# In[145]:


df_results_o


# ### Submission  RandomForest

# In[146]:


submission_randomF = submission.copy()
submission_randomF['SalePrice'] = randomF_predictions
submission_randomF['SalePrice'].head()


# In[147]:


# submission_randomF.to_csv('sample_submissionRandom.csv')


# ### Kaggle Score

# ![image.png](attachment:image.png)

# ### Evaluate Models

# - #### How can your model be used for inference? Why do you believe your model will generalize to new data? 
# According to the random forest important featrues, we inference those featrues can play a major part in prediction. As we got these result: train score : 0.9821625329876957 test score : 0.9821625329876957 And when we tested the Corss Validation of it, the results were: "[0.85885306, 0.80347532, 0.81402519, 0.77931117, 0.87881093]" With an average of: "0.8508149158677224" ~ 0.85 , this is a good ratio, as it means that the model can generalize any new data that can enter the model at 85 percent of accuracy, as this result indicates that the model is right fitt because a low viariance of it .

# - #### Consider your evaluation metrics score
# Our evaluation metrics scores are data cleaning and cross-validation
