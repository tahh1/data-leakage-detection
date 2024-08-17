#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression ,Ridge
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv")
df


# In[ ]:


cols={'W':'Win', 'R':'Runs_scored', 'AB' : 'At_bat' , 'H' : 'hit', '2B' : 'Double', '3B':'Triple',\
 'HR':'Home_runs','BB':'Base_on_balls', 'SO': 'Strike_out','SB':'Stolen_base','RA':'Run_average','ER':'Earned_runs', 'ERA':'Earned_runs_average','CG':'Complete_game','SHO':'Shoutout','SV':'Save','E':'Errors'}


# In[ ]:


df.rename(cols,axis=1,inplace=True)


# In[ ]:


df


# In[ ]:


df.shape


# # **PRIMARY INSPECTION**

# In[ ]:


df.describe()


# Using Describe method we have identify that mean and median have very less difference between each other so it means that the data (columns) are not skewed  That is : Our data is normally distributed

# **Observations:**
# 
# 1.  All the columns are numerical *values*
#    
# 
# 
# 
# 
# 

# In[ ]:


df.info()


# **Observation**
# 
# 2.   There are no string values and none of the columns has any missing values
# 
# 
# 

# In[ ]:


#checking for nulls
df.isnull().sum()


# # **2.EDA**

# In[ ]:


# first trying with Histplot
for i in df.columns:
  sns.histplot(df[i])
  plt.show()


# In[ ]:


# Now trying with  scatterplot
for i in df.columns:
  sns.scatterplot(x=df[i],y=df['Win'])
  plt.show()


# In[ ]:


#To measure its linearity
df1=df.corr()
df1


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df1)


# # **3.Feature engineering**

# In[ ]:


# so now we are filtering the value above 0.35 and - 0.35 
# because from the above step we get to know that multicollinearity exists

corre=df[list(df1[(df1['Win']>0.35) | (df1['Win']<-0.35)].index)]
corre.corr()


# In[ ]:


# removing the column with highest multicollinearity

m_colli=corre.drop(columns=['Run_average','Earned_runs','Shoutout'])
m_colli.columns


# In[ ]:


X=m_colli.drop(columns='Win')
y=m_colli['Win']


# # ***Removing Outliers***

# In[ ]:


# for i in X.columns:
#   X.loc[X[i]>X[i].quantile(0.90),i]=X[i].quantile(0.90)
#   X.loc[X[i]<X[i].quantile(0.10),i]=X[i].quantile(0.10)


# Tried Removing outliers and got a very bad score 
# 
#  Note : It is risk to remove outliers in such a small dataset it can lead to losing important data 

# # **4.Data Preprocessing**

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[ ]:


Sca=StandardScaler()
Sca.fit(X_train)
X_train=Sca.transform(X_train)
X_test=Sca.transform(X_test)


# # 5.Building models 

# # **Support Vector Machine**

# In[ ]:


sv=svm.SVR(kernel='linear',C=10,gamma=0.01)
sv.fit(X_train,y_train)


# In[ ]:


y_train_pred=sv.predict(X_train)
y_test_pred=sv.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# # ***Grid Search CV on SVM***

# In[ ]:


sv=svm.SVR(kernel='linear',C=40,gamma=0.001)
cross_val_score(sv,X_train,y_train,scoring='r2')


# In[ ]:


param_={'kernel':['poly','linear','rbf'],'C':[0.001,0.01,0.1,1,10,0.0001,20,50,80,100,120,150,200],'gamma':[0.01,0.1,1,10,0.001,0.0001]}
sv1=svm.SVR()
grid=GridSearchCV(sv1,param_grid=param_,cv=5,scoring='r2')
grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_


# In[ ]:


y_train_pred=grid.predict(X_train)
y_test_pred=grid.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


X_train.shape


# In[ ]:


sv=svm.SVR(kernel='linear',C=20, gamma=0.01,epsilon=1)
cross=cross_val_score(sv, X_train, y_train,scoring='r2', cv=3)
cross


# In[ ]:


#mean of validation score
np.mean(cross)


# In[ ]:


# standard deviation of validation scores
np.std(cross)


# # ***Decision Tree***

# In[ ]:


dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)


# In[ ]:


y_train_pred=dtr.predict(X_train)
y_test_pred=dtr.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# # Grid Search CV with decision Tree

# In[ ]:


param={'max_depth':[3,2,4,5,6,7,8,9],'min_samples_split':[1,2,3,4,5], 'min_impurity_split':[1,2,3,4,5],'max_features':[1,2,3,4,5,6]}
dt=DecisionTreeRegressor()
dtgrid=GridSearchCV(dt, param_grid=param,cv=3)
dtgrid.fit(X_train,y_train)


# In[ ]:


# checking the best parameters
dtgrid.best_params_


# In[ ]:


# using the best parameters for cross validation
dt1=DecisionTreeRegressor(max_depth=9, min_samples_split= 3)
cro=cross_val_score(dt1, X_train, y_train,scoring='r2')
cro


# In[ ]:


np.mean(cro)


# In[ ]:


np.std(cro)


# We can still see that there is still overfitting

# # ***XGBoost***

# In[ ]:


xb=xgb.XGBRegressor(n_estimators=100,min_child_weight=1)
xb.fit(X_train,y_train)


# In[ ]:


y_train_pred=xb.predict(X_train)
y_test_pred=xb.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# # XGboost with Grid Search CV

# In[ ]:


lr2=xgb.XGBRegressor(n_estimators=100,min_child_weight=1)


# In[ ]:


param={'booster':['gblinear','gbtree','dart'],'max_depth':[3,2,4,5,6,7,8,9],'min_child_weight':[1,2,3,4,5]}
xgb1=xgb.XGBRegressor()
xgb_grid=GridSearchCV(xgb1, param_grid=param,cv=5)
xgb_grid.fit(X_train,y_train)


# In[ ]:


y_train_pred=xgb_grid.predict(X_train)
y_test_pred=xgb_grid.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# In[ ]:


xgb_grid.best_params_


# In[ ]:


cross1=cross_val_score(lr2, X_train, y_train,scoring='r2')
cross1


# In[ ]:


np.mean(cross1)


# In[ ]:


np.std(cross1)


# # **Linear Regression**

# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


# This are the Coefficients

lr.coef_


# In[ ]:


y_train_pred=lr.predict(X_train)
y_test_pred=lr.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# In[ ]:


lr1=LinearRegression()


# In[ ]:


# checking the cross validation score
cr=cross_val_score(lr1,X_train,y_train,scoring='r2')
cr


# In[ ]:


#mean of cross validation score
np.mean(cr)


# In[ ]:


# std of validation scores
np.std(cr)


# # **Ridge Regression**

# In[ ]:


rid=Ridge()


# In[ ]:


rid.fit(X_train,y_train)


# In[ ]:


y_train_pred=rid.predict(X_train)
y_test_pred=rid.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# **Cross validation score**

# In[ ]:


rid1=Ridge()


# In[ ]:


# checking the cross validation score
Cro=cross_val_score(rid,X_train,y_train,scoring='r2')
Cro


# In[ ]:


#mean of cross validation score
np.mean(Cro)


# In[ ]:


# std of validation scores
np.std(Cro)


# # **Ridge Regression on Grid Search CV**

# In[ ]:


rid1=Ridge()


# In[ ]:


parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge1=GridSearchCV(rid1,parameters,scoring='r2',cv=5)


# In[ ]:


ridge1.fit(X_train,y_train)


# In[ ]:


y_train_pred=ridge1.predict(X_train)
y_test_pred=ridge1.predict(X_test)


# In[ ]:


r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# In[ ]:


ridge1.best_params_


# In[ ]:


Cr=cross_val_score(ridge1, X_train, y_train,scoring='r2')
Cr


# In[ ]:


np.mean(Cr)


# In[ ]:


np.std(Cr)


# According to cross validation score the best score is given by Linear Regression the mean is 81 and standard Deviation is less than other model

# # **After trying all the above models and hyperparameter tuning,Linear Regression gave some better results with the mean cross validation r2 score of 81% and the standard deviation of those scores being just 9%. R2 score on test set also is 81%.**

# # **Saving the Model**

# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


import pickle


# In[ ]:


# Save the trained model as a pickle string.
savemodel = pickle.dumps(lr)
 
# Load the pickled model
lr_from_pickle = pickle.loads(savemodel)
 
# Use the loaded pickled model to make predictions
lr_from_pickle.predict(X_test)


# In[ ]:





# In[ ]:





# In[ ]:




