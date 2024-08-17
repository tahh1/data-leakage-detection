#!/usr/bin/env python
# coding: utf-8

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

# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[92]:


df=pd.read_csv('Data/Real-Data/Real_Combine.csv')



# In[98]:


df.head()


# In[97]:


## Check for null values

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[95]:


df=df.dropna()


# In[99]:


X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features


# In[50]:


## check null values
X.isnull()


# In[51]:


y.isnull()


# In[ ]:





# In[ ]:





# In[52]:


sns.pairplot(df)


# In[53]:


df.corr()


# ### Correlation Matrix with Heatmap
# Correlation states how the features are related to each other or the target variable.
# 
# Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
# 
# Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

# In[54]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[55]:


corrmat.index


# ### Feature Importance
# You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Regressor, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset.

# In[100]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[104]:


X.head()


# In[105]:


print((model.feature_importances_))


# In[106]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# ### Linear Regression

# In[69]:


sns.distplot(y)


# ### Train Test split

# In[107]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[108]:


from sklearn.linear_model import LinearRegression


# In[109]:


regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[112]:


regressor.coef_


# In[113]:


regressor.intercept_


# In[110]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train))))


# In[111]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test))))


# In[77]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)


# In[79]:


score.mean()


# #### Model Evaluation

# In[114]:


coeff_df = pd.DataFrame(regressor.coef_,X.columns,columns=['Coefficient'])
coeff_df


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in T is associated with an *decrease of 2.690 in AQI PM2.5 *.
# - Holding all other features fixed, a 1 unit increase in TM is associated with an *increase of 0.46 in AQI PM 2.5 *.
# 

# In[115]:


prediction=regressor.predict(X_test)


# In[116]:


sns.distplot(y_test-prediction)


# In[88]:


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

# In[89]:


from sklearn import metrics


# In[91]:


print(('MAE:', metrics.mean_absolute_error(y_test, prediction)))
print(('MSE:', metrics.mean_squared_error(y_test, prediction)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction))))


# In[117]:


import pickle 


# In[118]:


# open a file, where you ant to store the data
file = open('regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)


# In[ ]:




