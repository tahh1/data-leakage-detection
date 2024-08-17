#!/usr/bin/env python
# coding: utf-8

# ## XgboostRegressor

# ### Apply ML algorithms
# 
# - Linear Regression
# - Lasso Regression
# - Decision Tree Regressor
# - KNN Regressor
# - RandomForestRegressor
# - Xgboost Regressor
# - Huperparameter Tuning
# - ANN- Artificial Neural Network

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Data/Real-Data/Real_Combine.csv')



# In[3]:


df.shape


# In[4]:


## Check for null values

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[5]:


df=df.dropna()


# In[6]:


X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features


# In[7]:


## check null values
X.isnull()


# In[8]:


y.isnull()


# In[ ]:





# In[ ]:





# In[9]:


sns.pairplot(df)


# In[11]:


df.corr()


# ### Correlation Matrix with Heatmap
# Correlation states how the features are related to each other or the target variable.
# 
# Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
# 
# Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

# In[11]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[12]:


corrmat.index


# ### Feature Importance
# You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Regressor, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset.

# In[13]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[14]:


X.head()


# In[15]:


print((model.feature_importances_))


# In[16]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# ### Linear Regression

# In[17]:


sns.distplot(y)


# ### Train Test split

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[15]:


import xgboost as xgb
#conda install -c ananconda py-xgboost


# In[16]:


regressor=xgb.XGBRegressor()
regressor.fit(X_train,y_train)


# In[17]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train))))


# In[18]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test))))


# In[19]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)


# In[20]:


score.mean()


# #### Model Evaluation

# In[21]:


prediction=regressor.predict(X_test)


# In[22]:


sns.distplot(y_test-prediction)


# In[23]:


plt.scatter(y_test,prediction)


# ## Hyperparameter Tuning

# In[ ]:


xgb.XGBRegressor()


# In[29]:


from sklearn.model_selection import RandomizedSearchCV


# In[51]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[25]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]



# In[26]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}

print(random_grid)


# In[27]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
regressor=xgb.XGBRegressor()


# In[30]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
xg_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[32]:


xg_random.fit(X_train,y_train)


# In[33]:


xg_random.best_params_


# In[34]:


xg_random.best_params_


# In[35]:


xg_random.best_score_


# In[39]:


rf_random.best_score_


# In[36]:


predictions=xg_random.predict(X_test)


# In[37]:


sns.distplot(y_test-predictions)


# In[39]:


plt.scatter(y_test,predictions)


# In[41]:


from sklearn import metrics
print(('MAE:', metrics.mean_absolute_error(y_test, predictions)))
print(('MSE:', metrics.mean_squared_error(y_test, predictions)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))))


# In[50]:


print(('MAE:', metrics.mean_absolute_error(y_test, predictions)))
print(('MSE:', metrics.mean_squared_error(y_test, predictions)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))))


# # Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[29]:


from sklearn import metrics


# In[30]:


print(('MAE:', metrics.mean_absolute_error(y_test, prediction)))
print(('MSE:', metrics.mean_squared_error(y_test, prediction)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction))))


# In[117]:


import pickle 


# In[52]:


# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[ ]:




