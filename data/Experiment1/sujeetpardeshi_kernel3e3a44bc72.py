#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, validation_curve
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 1
TEST_SIZE = .2
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
df.head()


# In[4]:


df = df.replace(np.nan,'',regex=True)


# In[5]:


df['Date'] = pd.to_datetime(df['Date'])


# # Convert datetime to days from 1-Jan-2020

# In[6]:


days = []
for index in df.index:
    if(df.iloc[index].Date.month == 1):
        days.append(df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 2):
        days.append((31) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 3):
        days.append((31+29) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 4):
        days.append((31*2+29) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 5):
        days.append((31*2+29+30) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 6):
        days.append((31*3+29+30*1) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 7):
        days.append((31*3+29+30*2) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 8):
        days.append((31*4+29+30*2) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 9):
        days.append((31*5+29+30*2) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 10):
        days.append((31*5+29+30*3) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 11):
        days.append((31*6+29+30*3) + df.iloc[index].Date.day)
    elif(df.iloc[index].Date.month == 12):
        days.append((31*6+29+30*4) + df.iloc[index].Date.day)
        
df.insert(0,'days', days)        
df.drop('Date', axis=1, inplace=True)


# # Merge Country region, Provinence state, country to single feature and apply binary encoding

# In[7]:


df['Region'] = df['Country_Region'] + df['Province_State'] + df['County']


# In[8]:


df.drop(['Id','Country_Region', 'Province_State', 'County'], inplace=True, axis=1)


# In[9]:


import category_encoders as ce
encoder = ce.BinaryEncoder(cols = ['Region'])
dfbin = encoder.fit_transform(df['Region'])
df = pd.concat([df, dfbin],axis=1)
df.drop('Region', axis=1, inplace=True)


# # Split covid dataset into 2 models confirmed cases and fatalities

# In[10]:


confirmCasesDf = df[df['Target'] == 'ConfirmedCases']
fatalitiesDf = df[df['Target'] == 'Fatalities']

confirmCasesDf.drop('Target', axis=1, inplace=True)
fatalitiesDf.drop('Target', axis=1, inplace=True)


# In[11]:


X = fatalitiesDf.drop('TargetValue', axis=1)
y = fatalitiesDf.TargetValue
xFatalities_train, xFatalities_test, yFatalities_train, yFatalities_test = train_test_split(X,
                                                                                            y,
                                                                                            test_size=TEST_SIZE,
                                                                                            random_state=RANDOM_STATE) 


# In[12]:


X = confirmCasesDf.drop('TargetValue', axis=1)
y = confirmCasesDf.TargetValue
xConfirmCases_train, xConfirmCases_test, yConfirmCases_train, yConfirmCases_test = train_test_split(X,
                                                                                                    y,
                                                                                                    test_size=TEST_SIZE,
                                                                                                    random_state=RANDOM_STATE) 


# # Create fatalities model

# In[13]:


fatalitiesModel = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, oob_score=True, n_estimators=300)
fatalitiesModel.fit(xFatalities_train, yFatalities_train)


# In[14]:


fatalitiesModel.oob_score_


# In[15]:


fatalitiesModel.score(xFatalities_train, yFatalities_train)


# In[16]:


fatalitiesModel.score(xFatalities_test, yFatalities_test)


# **#Create confirmed cases model**

# In[17]:


confirmedCaseModel = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, oob_score=True, n_estimators=300)
confirmedCaseModel.fit(xConfirmCases_train, yConfirmCases_train)


# In[18]:


confirmedCaseModel.oob_score_


# In[19]:


confirmedCaseModel.score(xConfirmCases_train, yConfirmCases_train)


# In[20]:


confirmedCaseModel.score(xConfirmCases_test, yConfirmCases_test)


# # ###Load test data

# In[30]:


testDf = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
testDf.head()


# # Do preprocessing on Date, County, Provinence_state, Country region 

# In[31]:


testDf['Date'] = pd.to_datetime(testDf['Date'])


# In[32]:


testDf = testDf.replace(np.nan,'',regex=True)


# In[33]:


days = []
for index in testDf.index:
    if(testDf.iloc[index].Date.month == 1):
        days.append(testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 2):
        days.append((31) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 3):
        days.append((31+29) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 4):
        days.append((31*2+29) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 5):
        days.append((31*2+29+30) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 6):
        days.append((31*3+29+30*1) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 7):
        days.append((31*3+29+30*2) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 8):
        days.append((31*4+29+30*2) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 9):
        days.append((31*5+29+30*2) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 10):
        days.append((31*5+29+30*3) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 11):
        days.append((31*6+29+30*3) + testDf.iloc[index].Date.day)
    elif(testDf.iloc[index].Date.month == 12):
        days.append((31*6+29+30*4) + testDf.iloc[index].Date.day)
        
testDf.insert(0,'days', days)        
testDf.drop('Date', axis=1, inplace=True)


# In[34]:


testDf['Region'] = testDf['Country_Region'] + testDf['Province_State'] + testDf['County']


# In[35]:


testDf.drop(['Country_Region', 'Province_State', 'County'], inplace=True, axis=1)


# In[36]:


import category_encoders as ce
encoder = ce.BinaryEncoder(cols = ['Region'])
dfbin = encoder.fit_transform(testDf['Region'])
testDf = pd.concat([testDf, dfbin],axis=1)
testDf.drop('Region', axis=1, inplace=True)


# In[37]:


testConfirmCasesDf = testDf[testDf['Target'] == 'ConfirmedCases']
testFatalitiesDf = testDf[testDf['Target'] == 'Fatalities']

confirmedCasesForecastIds = testConfirmCasesDf.ForecastId
fatilitiesForecastIds = testFatalitiesDf.ForecastId

testConfirmCasesDf.drop(['Target', 'ForecastId'], axis=1, inplace=True)
testFatalitiesDf.drop(['Target', 'ForecastId'], axis=1, inplace=True)


# # Predict using fatalities and confirmed cases model

# In[45]:


confirmTargetScore = confirmedCaseModel.predict(testConfirmCasesDf)


# In[46]:


fatalitiesScore = fatalitiesModel.predict(testFatalitiesDf)


# In[47]:


confirmTargetDict = dict(list(zip(confirmedCasesForecastIds, confirmTargetScore)))
fatalitiesDict = dict(list(zip(fatilitiesForecastIds, fatalitiesScore)))

finalDict = { **confirmTargetDict, **fatalitiesDict }


# In[56]:


resultDf = pd.DataFrame({"ForecastId":list(finalDict.keys()), 'TargetValue':list(finalDict.values())})
resultDf.sort_values(by=['ForecastId'], inplace=True)
resultDf.head()


# In[52]:





# In[57]:


a=resultDf.groupby(['ForecastId'])['TargetValue'].quantile(q=0.05).reset_index()
b=resultDf.groupby(['ForecastId'])['TargetValue'].quantile(q=0.5).reset_index()
c=resultDf.groupby(['ForecastId'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['ForecastId','q0.05']
b.columns=['ForecastId','q0.5']
c.columns=['ForecastId','q0.95']
a=pd.concat([a,b['q0.5'],c['q0.95']],1)
a['q0.05']=a['q0.05'].clip(0,10000)
a['q0.5']=a['q0.5'].clip(0,10000)
a['q0.95']=a['q0.95'].clip(0,10000)
a


# In[58]:


sub=pd.melt(a, id_vars=['ForecastId'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['ForecastId'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.head()


# # Save final result

# In[60]:


sub.to_csv("submission.csv",index=False)


# In[ ]:




