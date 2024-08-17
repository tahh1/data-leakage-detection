#!/usr/bin/env python
# coding: utf-8

# ## Artificial Neural Network

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


# In[3]:


## Check for null values

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[4]:


df=df.dropna()


# In[5]:


X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features


# In[6]:


## check null values
X.isnull()


# In[7]:


y.isnull()


# In[ ]:





# In[ ]:





# In[9]:


sns.pairplot(df)


# In[8]:


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


# ### ANN

# In[17]:


sns.distplot(y)


# ### Train Test split

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[11]:


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[15]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

# Fitting the ANN to the Training set
model_history=NN_model.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)


# #### Model Evaluation

# In[18]:


prediction=NN_model.predict(X_test)


# In[26]:


y_test


# In[27]:


sns.distplot(y_test.values.reshape(-1,1)-prediction)


# In[28]:


plt.scatter(y_test,prediction)


# In[30]:


from sklearn import metrics
print(('MAE:', metrics.mean_absolute_error(y_test, prediction)))
print(('MSE:', metrics.mean_squared_error(y_test, prediction)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction))))


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




