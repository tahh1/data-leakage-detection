#!/usr/bin/env python
# coding: utf-8

# In[1]:


# My forecasting COVID-19 confirmed cases and fatalities between March 19 and April 30 
# My submission scored 0.53110

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# model
from catboost import Pool
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

#plot
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))


# In[2]:


# load training and testing data 
subm = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv', index_col='Id', parse_dates=True)
test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv', index_col='ForecastId', parse_dates=True)


# In[3]:


subm


# In[4]:


# see testing data
test_data


# In[5]:


# ...and training data
train_data


# In[6]:


train_data.describe()


# In[7]:


train_data.describe(include=['O'])


# In[8]:


test_data.describe()


# In[9]:


test_data.describe(include=['O'])


# In[10]:


train_data.shape


# In[11]:


test_data.shape


# In[12]:


# detect missing values in training
train_data.isna().sum()


# In[13]:


# ...in testing data
test_data.isna().sum()


# In[14]:


# Count number of unique elements in the train data
train_data.nunique()


# In[15]:


# Count number of unique elements in the test data
test_data.nunique()


# In[16]:


#Convert data in integer
train_data['Date']= pd.to_datetime(train_data['Date']).dt.strftime("%m%d").astype(int)
test_data['Date']= pd.to_datetime(test_data['Date']).dt.strftime("%m%d").astype(int)


# In[17]:


train_data.describe()


# In[18]:


# separate the vector correct answers ('ConfirmedCases' and 'Fatalities') from the training data
train_data.dropna(axis=0, subset=['ConfirmedCases', 'Fatalities'], inplace=True)
y_conf = train_data.ConfirmedCases
train_data.drop(['ConfirmedCases'], axis=1, inplace=True)
y_fatal = train_data.Fatalities
train_data.drop(['Fatalities'], axis=1, inplace=True)


# In[19]:


# Select categorical columns in training and testing data
categorical_cols = [cname for cname in train_data.columns if
                    train_data[cname].dtype == "object"]


# In[20]:


# replace missing values in training and testing data
# as we saw above, the data are absent only in 'Province/State'
train_data.fillna('-', inplace=True)
test_data.fillna('-',inplace=True)


# In[21]:


train_data.shape


# In[22]:


# perform LabelEncoder with categorical data (categorical_cols)
encodering = LabelEncoder()

encod_train_data = train_data.copy()
encod_test_data = test_data.copy()
for col in categorical_cols:
    encod_train_data[col] = encodering.fit_transform(train_data[col])
    encod_test_data[col] = encodering.fit_transform(test_data[col])


# In[23]:


# split encod_train_data into training(X_train) and validation(X_valid) data
# and split vector correct answers ('ConfirmedCases')
X_train, X_valid, y_train, y_valid = train_test_split(encod_train_data, y_conf, train_size=0.8, 
                                                      test_size=0.2, random_state=0)


# In[24]:


# determine the best metrics for the model
def get_score(n_estimators):
    model = GradientBoostingRegressor(n_estimators=n_estimators)
    scores = cross_val_score(model, X_train, y_train, cv=5)

    return scores.mean()


# In[25]:


def rmse_score(early_stopping_rounds):
    rmse = np.sqrt(-cross_val_score(CatBoostRegressor(iterations=4000, 
                                                      depth=9, 
                                                      learning_rate=0.5, 
                                                      loss_function='RMSE',
                                                      early_stopping_rounds = early_stopping_rounds,
                                                      verbose=False),
                                    X_train, y_train, scoring="neg_mean_squared_error", cv = 10))
    return(rmse)


# In[26]:


#metrics = [4, 2]
#results = {}
#for x in metrics:
    #results[x] = rmse_score(x)


# In[27]:


#for x in metrics:
    #print(x, results[x].mean())


# In[28]:


#results


# In[29]:


#plt.figure(figsize=(12,8))
#for i in results:
    #sns.lineplot(data=results[i], label=i)


# In[30]:


# select model and install parameters
model = CatBoostRegressor(iterations=4000, 
                          depth=9, 
                          learning_rate=0.2, 
                          loss_function='RMSE',
                          verbose=False)


# In[31]:


# train the model
model.fit(X_train,y_train)


# In[32]:


# preprocessing of validation data, get predictions
preds = model.predict(X_valid)

print(('MAE:', mean_absolute_error(y_valid, preds)))


# In[33]:


# make the prediction using the resulting model
preds = model.predict(X_valid)

print(('MSE:', mean_squared_error(y_valid, preds)))


# In[34]:


x_list = [X_train, X_valid]
y_list = [y_train, y_valid]

scoring = list(map(lambda x,y: round(model.score(x,y)*100, 2), x_list, y_list)) 
scoring


# In[35]:


# get predictions test data
final_preds_conf = model.predict(encod_test_data)


# In[36]:


# split encod_train_data into training(X_train) and validation(X_valid) data
# and split vector correct answers ('Fatalities')
X_train_f, X_valid_f, y_train_f, y_valid_f = train_test_split(encod_train_data, y_fatal, train_size=0.8, 
                                                      test_size=0.2, random_state=0)


# In[37]:


# train the model
model.fit(X_train_f,y_train_f)


# In[38]:


# preprocessing of validation data, get predictions
preds = model.predict(X_valid_f)

print(('MAE:', mean_absolute_error(y_valid_f, preds)))


# In[39]:


# make the prediction using the resulting model
preds = model.predict(X_valid_f)

print(('MSE:', mean_squared_error(y_valid_f, preds)))


# In[40]:


x_list_f = [X_train_f, X_valid_f]
y_list_f = [y_train_f, y_valid_f]

scoring = list(map(lambda x,y: round(model.score(x,y)*100, 2), x_list_f, y_list_f)) 
scoring


# In[41]:


# get predictions test data
final_preds_fatal = model.predict(encod_test_data)


# In[42]:


# combine predictions 'ConfirmedCases' and 'Fatalities'
output = pd.DataFrame({'ForecastId': test_data.index,
                       'ConfirmedCases': final_preds_conf,
                       'Fatalities': final_preds_fatal})


# In[43]:


# replace negative values with 0, because the predictions of 'ConfirmedCases' and 'Fatalities' cannot be negative
output.loc[output['ConfirmedCases'] < 0,'ConfirmedCases'] = 0
output.loc[output['Fatalities'] < 0,'Fatalities'] = 0


# In[44]:


# and save test predictions to file
output.to_csv('submission.csv', index=False)
print('Complete!')


# In[45]:


output.tail(30)


# In[46]:


output.describe()


# In[47]:


plt.figure(figsize=(12,8))
sns.lineplot(data=output['ConfirmedCases'], label="ConfirmedCases")
sns.lineplot(data=output['Fatalities'], label="Fatalities")


# In[48]:


plt.figure(figsize=(12,8))
sns.scatterplot(x=output['ForecastId'], y=output['ConfirmedCases'], hue=output['Fatalities'])

