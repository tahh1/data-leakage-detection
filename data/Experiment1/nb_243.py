#!/usr/bin/env python
# coding: utf-8

# In[212]:


# neccessary imports
import pandas as pd
import numpy as np


# In[213]:


# reading the data
data=pd.read_csv('../data/insuranceFraud.csv')


# In[214]:


# Having a look at the data
data.head()


# In[215]:


# In this dataset missing values have been denoted by '?'
# we are replacing ? with NaN for them to be imputed down the line.
data=data.replace('?',np.nan)


# In[216]:


# list of columns not necessary for pfrediction
cols_to_drop=['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']


# In[217]:


# dropping the unnecessary columns
data.drop(columns=cols_to_drop,inplace=True)


# In[218]:


# checking the data after dropping the columns
data.head()


# In[219]:


# checking for missing values
data.isna().sum()


# In[220]:


# checking for th number of categorical and numerical columns
data.info()


# In[221]:


# As the columns which have missing values, they are only categorical, we'll use the categorical imputer
# Importing the categorical imputer
from sklearn_pandas import CategoricalImputer
imputer = CategoricalImputer()


# In[222]:


# imputing the missing values from the column

data['collision_type']=imputer.fit_transform(data['collision_type'])
data['property_damage']=imputer.fit_transform(data['property_damage'])
data['police_report_available']=imputer.fit_transform(data['police_report_available'])


# In[223]:


# Extracting the categorical columns
cat_df = data.select_dtypes(include=['object']).copy()


# In[224]:


cat_df.columns


# In[225]:


cat_df.head()


# Checking the categorical values present in the columns to decide for getDummies encode or custom mapping to convert categorical data to numeric one

# In[226]:


cat_df.columns


# In[227]:


cat_df['policy_csl'].unique()


# In[228]:


cat_df['insured_education_level'].unique()


# In[229]:


cat_df['incident_severity'].unique()


# In[99]:


#cat_df['property_damage'].unique()


# In[230]:


# custom mapping for encoding
cat_df['policy_csl'] = cat_df['policy_csl'].map({'100/300' : 1, '250/500' : 2.5 ,'500/1000':5})
cat_df['insured_education_level'] = cat_df['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
cat_df['incident_severity'] = cat_df['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
cat_df['insured_sex'] = cat_df['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
cat_df['property_damage'] = cat_df['property_damage'].map({'NO' : 0, 'YES' : 1})
cat_df['police_report_available'] = cat_df['police_report_available'].map({'NO' : 0, 'YES' : 1})
cat_df['fraud_reported'] = cat_df['fraud_reported'].map({'N' : 0, 'Y' : 1})


# In[231]:


# auto encoding of categorical variables
for col in cat_df.drop(columns=['policy_csl','insured_education_level','incident_severity','insured_sex','property_damage','police_report_available','fraud_reported']).columns:
    cat_df= pd.get_dummies(cat_df, columns=[col], prefix = [col], drop_first=True)


# In[232]:


# data fter encoding
cat_df.head()


# In[233]:


# extracting the numerical columns
num_df = data.select_dtypes(include=['int64']).copy()


# In[234]:


num_df.columns


# In[235]:


num_df.head()


# In[236]:


# combining the Numerical and categorical dataframes to get the final dataset
final_df=pd.concat([num_df,cat_df], axis=1)


# In[237]:


final_df.head()


# In[279]:


# separating the feature and target columns
x=final_df.drop('fraud_reported',axis=1)
y=final_df['fraud_reported']


# In[210]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', ' inline')


# In[122]:


# we'll look at the distribution of data in some columns now
plt.hist(final_df['policy_csl'])
# we  can see that for almost all categories of CSL the data is uniformly distributed


# In[130]:


import seaborn as sns


# In[132]:


sns.distplot(final_df['insured_sex'])
# we  can see that for almost all categories of the gender of the insured the data is uniformly distributed


# In[133]:


sns.distplot(final_df['insured_education_level'])
# we  can see that for almost all categories of the education level of the person insured the data is uniformly distributed


# In[134]:


sns.distplot(final_df['incident_severity'])
"""
We can see that there are least claims for trivial incidents,
most claims for minor incidents,
and for major and Total loss incidents the claims are almost equal.
"""


# In[135]:


num_df.columns


# In[137]:


sns.scatterplot(final_df['months_as_customer'],final_df['age'], hue=final_df['fraud_reported'] )
"""
from the graph it can be concluded that most of the fraud cases are done by the customers new 
to the company and that too comparatively younger ones. 
"""


# In[143]:


plt.figure(figsize=(13,8))
sns.heatmap(num_df.corr(), annot=True )


# From the plot above, we can see that there is high correlation between Age and the number of months. we'll drop the age column.
# Also, there is high correlation between total claim amount, injury claim,vehicle claim, and property claim as total claim is the sum of all others. So, we'll drop the total claim column.

# In[239]:


x.columns


# In[280]:


x.drop(columns=['age','total_claim_amount'], inplace=True)


# In[294]:


# splitting the data for model training

# splitting the data into training and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y, random_state=355 )


# In[295]:


train_x.head()


# In[296]:


num_df=train_x[['months_as_customer', 'policy_deductable', 'umbrella_limit',
       'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
       'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim', 'property_claim',
       'vehicle_claim']]


# In[297]:


num_df.columns


# In[298]:


print((train_x.shape))
print((num_df.shape))


# In[299]:


# Scaling the numeric values in the dataset

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[300]:


scaled_data=scaler.fit_transform(num_df)
scaled_num_df= pd.DataFrame(data=scaled_data, columns=num_df.columns,index=train_x.index)
scaled_num_df.shape


# In[301]:


scaled_num_df.isna().sum()


# In[302]:


train_x.drop(columns=scaled_num_df.columns, inplace=True)


# In[303]:


train_x.shape


# In[269]:


train_x.head()


# In[304]:


train_x=pd.concat([scaled_num_df,train_x],axis=1)


# In[292]:


#train_x[:20]


# In[306]:


#train_x.isna().sum()


# In[307]:


# first using the Support vector classifier for model training
from sklearn.svm import SVC
sv_classifier=SVC()


# In[308]:


y_pred = sv_classifier.fit(train_x, train_y).predict(test_x)


# In[309]:


from sklearn.metrics import accuracy_score


# In[310]:


sc=accuracy_score(test_y,y_pred)
sc


# In[311]:


from sklearn.model_selection import GridSearchCV


# In[314]:


param_grid = {"kernel": ['rbf','sigmoid'],
             "C":[0.1,0.5,1.0],
             "random_state":[0,100,200,300]}


# In[316]:


grid = GridSearchCV(estimator=sv_classifier, param_grid=param_grid, cv=5,  verbose=3)


# In[317]:


grid.fit(train_x, train_y)


# In[324]:


grid.best_estimator_


# In[331]:


from xgboost import XGBClassifier


# In[341]:


xgb=XGBClassifier()


# In[342]:


y_pred = xgb.fit(train_x, train_y).predict(test_x)


# In[343]:


ac2=accuracy_score(test_y,y_pred)
ac2


# In[335]:


param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": list(range(2, 10, 1))}

            #Creating an object of the Grid Search class
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5,  verbose=3,n_jobs=-1)


# In[336]:


#finding the best parameters
grid.fit(train_x, train_y)


# In[337]:


grid.best_estimator_


# In[ ]:




