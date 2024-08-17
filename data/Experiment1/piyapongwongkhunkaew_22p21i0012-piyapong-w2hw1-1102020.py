#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


PROJECT_ID = 'your-google-cloud-project'
from google.cloud import automl_v1beta1 as automl
automl_client = automl.AutoMlClient()
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
from google.cloud import bigquery
bigquery_client = bigquery.Client(project=PROJECT_ID)


# Q1. ให้ใช้ข้อมูลที่อยู่ใน Plant_1_Generation_Data.csv และ Plant_1_Weather_Sensor_Data.csv
# 
#     a. แบ่งข้อมูลออกเป็น 2 ชุด (1) training 90%, (2) testing 10% โดยกำหนดแบ่ง 10 fold ตามนี้ คือ
#         แถวที่ 1, 11, 21, 31, … เป็นเซ็ตที่ 1
#         แถวที่ 2, 12, 22, 32, … เป็นเซ็ตที่ 2
#         แถวที่ 3, 13, 23, 33, … เป็นเซ็ตที่ 3
#         …..
#         แถวที่ 10, 20, 30, 40, … เป็นเซ็ตที่ 10

# In[3]:


# เรียกข้อมูล
dfG = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)
dfW = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)


# In[4]:


dfG


# In[5]:


# จัดการข้อมูล
dfG = dfG.groupby('DATE_TIME').agg({
    'DC_POWER': 'mean',
    'AC_POWER': 'mean',
    'DAILY_YIELD': 'mean',
    'TOTAL_YIELD': 'mean',
}).reset_index()
dfG


# In[6]:


dfG
dfG.describe()
dfG.info()


# In[7]:


dfG.head()


# In[8]:


dfW
dfW.describe()
dfW.info()


# In[9]:


# Selected Feature

dfW = dfW.drop(['PLANT_ID','SOURCE_KEY'], axis='columns')


# In[10]:


dfW


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

x = dfG['DATE_TIME']
y1 = dfG['DC_POWER']
y2 = dfG['AC_POWER']
y3 = dfG['DAILY_YIELD']
y4 = dfG['TOTAL_YIELD']


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
fig.suptitle('Generation_Data')

ax1.plot(x, y1, 'r.-')
ax1.set_ylabel('DC_POWER')

ax2.plot(x, y2, 'b.-')
ax2.set_xlabel('time')
ax2.set_ylabel('AC_POWER')

ax3.plot(x, y3, 'g.-')
ax3.set_ylabel('DAILY_YIELD')

ax4.plot(x, y4, 'y.-')
ax4.set_xlabel('time')
ax4.set_ylabel('TOTAL_YIELD')

plt.show()


# In[12]:


dfG.head()


# In[13]:


X1 = dfG.iloc[:, [1,2,3]]
y1 = dfG.iloc[:, 4]


# In[14]:


X1, y1


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X1 = scaler.fit_transform(X1)


# In[16]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import metrics


# In[17]:


def KFoldScore(reg, X1, y1, cv=10):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X1)
    
    RMSE = []
    
    for train_idx, test_idx in kf.split(X1):
        X_train = X1[train_idx]
        X_test = X1[test_idx]
        y_train = y1[train_idx]
        y_test = y1[test_idx]
        
        reg.fit(X_train, y_train)
        y_pred = np.round(reg.predict(X_test))
             
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        RMSE.append(rmse)
        
    return np.mean(RMSE)


# In[18]:


# 1.1.1. Ordinary Least Squares¶
ols = linear_model.LinearRegression()
# 1.1.2. Ridge regression
rr = linear_model.Ridge(alpha=.5)
# 1.1.3. Lasso
l = linear_model.Lasso(alpha=0.1)
# 1.1.4 LARS Lasso
ll = linear_model.LassoLars(alpha=.1)
#1.1.10.1. Bayesian Ridge Regression
brr = linear_model.BayesianRidge()
# 1.1.12. Generalized Linear Regression
glr = TweedieRegressor(power=1, alpha=0.5, link='log')
# multiple Linear Regression
mlp = MLPRegressor(max_iter=500)


# In[19]:


ols_error = KFoldScore(ols, X1, y1, cv=10)
rr_error  = KFoldScore(rr, X1, y1, cv=10)
l_error  = KFoldScore(l, X1, y1, cv=10)
ll_error  = KFoldScore(ll, X1, y1, cv=10)
brr_error  = KFoldScore(brr, X1, y1, cv=10)
glr_error  = KFoldScore(glr, X1, y1, cv=10)
mlp_error  = KFoldScore(mlp, X1, y1, cv=10)


# In[20]:


print(f'Ordinary Least Squares RMSE: {ols_error}\nRidge regression RMSE: {rr_error}\nLasso RMSE: {l_error}\nLARS Lasso RMSE: {ll_error}\nBayesian Ridge Regression RMSE: {brr_error}\nGeneralized Linear Regression RMSE: {glr_error}\nMultiple Linear Regression RMSE: {mlp_error}')

