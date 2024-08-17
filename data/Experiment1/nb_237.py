#!/usr/bin/env python
# coding: utf-8

# ## Decision Tree Regressor Air Quality Index Prediction

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

# In[145]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[146]:


df=pd.read_csv('Data/Real-Data/Real_Combine.csv')



# In[147]:


df.head()


# In[148]:


## Check for null values

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[149]:


df=df.dropna()


# In[150]:


X=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features


# In[151]:


## check null values
X.isnull()


# In[152]:


y.isnull()


# In[ ]:





# In[ ]:





# In[155]:


sns.pairplot(df)


# In[156]:


df.corr()


# ### Correlation Matrix with Heatmap
# Correlation states how the features are related to each other or the target variable.
# 
# Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
# 
# Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

# In[157]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[158]:


corrmat.index


# ### Feature Importance
# You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
# 
# Feature importance gives you a score for each feature of your data, the higher the score more important or relevant is the feature towards your output variable.
# 
# Feature importance is an inbuilt class that comes with Tree Based Regressor, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset.

# In[159]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[160]:


X.head()


# In[161]:


print((model.feature_importances_))


# In[162]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# ### Decision Tree Regressor

# In[163]:


sns.distplot(y)


# ### Train Test split

# In[164]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[10]:


from sklearn.tree import DecisionTreeRegressor


# In[165]:


dtree=DecisionTreeRegressor(criterion="mse")


# In[166]:


dtree.fit(X_train,y_train)


# In[168]:


print(("Coefficient of determination R^2 <-- on train set: {}".format(dtree.score(X_train, y_train))))


# In[170]:


print(("Coefficient of determination R^2 <-- on test set: {}".format(dtree.score(X_test, y_test))))


# In[172]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(dtree,X,y,cv=5)


# In[173]:


score.mean()


# ## Tree Visualization
# 
# Scikit learn actually has some built-in visualization capabilities for decision trees, you won't use this often and it requires you to install the pydot library, but here is an example of what it looks like and the code to execute this:

# In[175]:


##conda install pydotplus
## conda install python-graphviz

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus



# In[176]:


features = list(df.columns[:-1])
features


# In[177]:


import os

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


# In[22]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[21]:





# #### Model Evaluation

# In[23]:


prediction=dtree.predict(X_test)


# In[24]:


sns.distplot(y_test-prediction)


# In[178]:


plt.scatter(y_test,prediction)


# ### Hyperparameter Tuning DEcision Tree Regressor

# In[ ]:


DecisionTreeRegressor()


# In[121]:


## Hyper Parameter Optimization

params={
 "splitter"    : ["best","random"] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_samples_leaf" : [ 1,2,3,4,5 ],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
 "max_features" : ["auto","log2","sqrt",None ],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70]
    
}


# In[122]:


## Hyperparameter optimization using GridSearchCV
from sklearn.model_selection import GridSearchCV


# In[126]:


random_search=GridSearchCV(dtree,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)


# In[127]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2))))


# In[128]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable


# In[129]:


random_search.best_params_


# In[130]:


random_search.best_score_


# In[131]:


predictions=random_search.predict(X_test)


# In[132]:


sns.distplot(y_test-predictions)


# In[ ]:





# In[134]:


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

# In[26]:


from sklearn import metrics


# In[27]:


print(('MAE:', metrics.mean_absolute_error(y_test, prediction)))
print(('MSE:', metrics.mean_squared_error(y_test, prediction)))
print(('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction))))


# In[117]:


import pickle 


# In[179]:


# open a file, where you ant to store the data
file = open('decision_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(random_search, file)


# In[ ]:




