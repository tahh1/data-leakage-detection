#!/usr/bin/env python
# coding: utf-8

# ## Ridge and Lasso Regression

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


df.head()


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


# In[10]:


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

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# ### Comparison Linear, Ridge and Lasso Regression

# ## Linear Regression

# In[45]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


# In[47]:


lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,X,y,scoring='neg_mean_squared_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)


# ## Ridge Regression

# In[55]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[64]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)


# In[65]:


print((ridge_regressor.best_params_))
print((ridge_regressor.best_score_))


# ## Lasso Regression

# In[61]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[66]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X,y)
print((lasso_regressor.best_params_))
print((lasso_regressor.best_score_))


# #### Model Evaluation

# In[37]:


prediction=lasso_regressor.predict(X_test)


# In[38]:


sns.distplot(y_test-prediction)


# In[39]:


plt.scatter(y_test,prediction)


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

# In[40]:


from sklearn import metrics


# In[67]:


print(('MAE:', metrics.mean_absolute_error(y_test, prediction)))
print(('MSE:', metrics.mean_squared_error(y_test, prediction)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction))))


# In[42]:


import pickle 


# In[44]:


# open a file, where you ant to store the data
file = open('lasso_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(lasso_regressor, file)


# In[ ]:




