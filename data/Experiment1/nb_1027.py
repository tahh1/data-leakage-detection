#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[37]:


from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0


# In[38]:


get_ipython().system('conda install -c anaconda xlrd --yes')


# In[39]:


# The code was removed by Watson Studio for sharing.


# In[40]:


df_data_1.describe()


# In[41]:


df=df_data_1[['SEVERITYCODE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','ROADCOND','LIGHTCOND','WEATHER','INCDTTM']]
df.head(9)


# In[42]:


df.dtypes


# In[43]:


#dropping empty rows
df1=df.dropna()
df1.head()


# In[104]:


# Select wanted columns
df1=df1[['SEVERITYCODE','PERSONCOUNT','PEDCOUNT','PEDCYLCOUNT','VEHCOUNT','ROADCOND','LIGHTCOND','WEATHER']]
df1.head(9)


# In[105]:


df1.shape


# In[108]:


#define labels
y = df1['SEVERITYCODE']
X = df1.drop('SEVERITYCODE', axis=1)
df1.head()
#Normalize data
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[109]:


LABELS= ['1','2']
count_codes = pd.value_counts(df1['SEVERITYCODE'], sort = True)
count_codes.plot(kind = 'bar', rot=0)
plt.xticks(list(range(2)), LABELS)
plt.title("Observation Frequency")
plt.xlabel("SEVERITYCODE")
plt.ylabel("Number of Observations");


# In[116]:


#Testing basemodel with Logistic Regression

#Set test training
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
 
#function
def run_model(X_train, X_test, y_train, y_test):
    clf_base = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg")
    clf_base.fit(X_train, y_train)
    return clf_base
 #executing basemodel
model = run_model(X_train, X_test, y_train, y_test)
 
#define work
def showresults(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True SeverityCode')
    plt.xlabel('Predicted SeverityCode')
    plt.show()
    print((classification_report(y_test, pred_y)))
 
pred_y = model.predict(X_test)
showresults(y_test, pred_y)


# In[117]:


#Improve recall=0.22 which is too slow

#Applied weight = “balanced” to our training model
def run_model_balanced(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1.0,penalty='l2',random_state=1,solver="newton-cg",class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf
 
model = run_model_balanced(X_train, X_test, y_train, y_test)
pred_y = model.predict(X_test)
showresults(y_test, pred_y)

