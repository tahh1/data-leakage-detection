#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Classification template
# (Cleaned up version)

# Logistic regression on Social Network dataset
# by taking into account the age and estimated salary to predict the output

# The Social Network dataset contains 400 entries and 3 relevant features:
# gender, age, estimated salary (and if the user purchased)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from helpers import plot_tools as ptools


# In[10]:


# Importing the dataset
dataset = pd.read_csv('../Datasets/Social_Network_Ads.csv') #,header=None)
X = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

classes=('red', 'blue')


# In[11]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[12]:


# Feature Scaling
# https://datascience.stackexchange.com/a/12346

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[13]:


# Fitting classifier to the Training set
# Create your classifier here

classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[14]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

cm = confusion_matrix(y_test, y_pred)
print((
    'vrais positifs \t faux négatifs', '\n'
    'faux positifs \t vrais négatifs', '\n'
))
print(cm)


# In[20]:


# Visualising the Training set results
ptools.plot_decision_regions_OG(X_train, y_train, 'Training set', 'Age', 'Estimated Salary', model=classifier)
ptools.plot_decision_regions_OG(X_test, y_test, 'Test set', 'Age', 'Estimated Salary', model=classifier)

