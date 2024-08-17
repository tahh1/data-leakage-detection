#!/usr/bin/env python
# coding: utf-8

# ## KNNRegressor

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


# ### K Nearest Neighbor Regression

# In[17]:


sns.distplot(y)


# ### Train Test split

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[13]:


from sklearn.neighbors import KNeighborsRegressor


# In[14]:


regressor=KNeighborsRegressor(n_neighbors=1)
regressor.fit(X_train,y_train)


# In[15]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train))))


# In[16]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test))))


# In[17]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(regressor,X,y,cv=5)


# In[18]:


score.mean()


# #### Model Evaluation

# In[19]:


prediction=regressor.predict(X_test)


# In[20]:


sns.distplot(y_test-prediction)


# In[21]:


plt.scatter(y_test,prediction)


# ## Hyperparameter Tuning

# In[27]:


accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsRegressor(n_neighbors=i)
    score=cross_val_score(knn,X,y,cv=10,scoring="neg_mean_squared_error")
    accuracy_rate.append(score.mean())


# In[28]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(list(range(1,40)),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
#plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
 #        markerfacecolor='red', markersize=10)
plt.title('Accuracy Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Rate')


# In[31]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)


# In[32]:


sns.distplot(y_test-predictions)


# In[33]:


plt.scatter(y_test,predictions)


# In[34]:


from sklearn import metrics
print(('MAE:', metrics.mean_absolute_error(y_test, predictions)))
print(('MSE:', metrics.mean_squared_error(y_test, predictions)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))))


# In[55]:


# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train,y_train)
predictions = knn.predict(X_test)


# In[52]:


sns.distplot(y_test-predictions)


# In[53]:


plt.scatter(y_test,predictions)


# In[56]:


from sklearn import metrics
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




