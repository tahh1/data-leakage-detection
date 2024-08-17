#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hello people, welcome to my kernel! In this kernel I am going to classify Mushrooms edible or poisonous. In this kernel I am going to use SKLearn for machine learning, Plotly for visualization and Numpy,Pandas for manipulating.
# 
# Before the start, let's take a look at our schedule
# 
# # Schedule
# 1. Importing Libraries and Data
# 1. Data Overview
# 1. Data Preprocessing
# 1. Feature Engineering
#     * Applying ***get_dummies*** Function For Each Function
# 1. Modeling
#     1. Classification Algorithms
#         * Support Vector Machine Classification
#         * Decision Tree Classification
#         * Naive Bayes Classification
#         * Random Forest Classification
# 1. Result | Best Algorithm for Mushroom Classification
# 1. Conclusion

# # Importing Libraries and The Data
# 
# In this section I am going to import libraries and the data that I will use. In the introduction section I've said which libraries that I will use, but in this section I am going to repeat them.
# 
# * For machine learning: SKLearn
# * For visualization: Plotly, missingno
# * For data manipulating : Pandas and Numpy

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

"""
Data Manipulating
"""
import numpy as np 
import pandas as pd 


"""
Visualization
"""
import plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# I am going to import sklearn modules when I need them so I did not import them

# In[2]:


data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')


# # Data Overview
# 
# In this section I am going to take an idea about data. In order to do this I am going to use head(),info(),isnull() methods.

# In[3]:


data.head()


# * There are 23 columns in data. 
# * All of the columns are category.

# In[4]:


data.info()


# * There are 8124 rows in the dataset.
# * All of the dataset is object (However we will convert them into category)

# In[5]:


data.isnull().sum()


# * There is no missing values in the dataset.

# # Data Preprocessing
# 
# In this section I am going to convert features into category. In order to do this I am going to define a function that helps us.

# In[6]:


def converter(df,features):
    """
    A function that converts features into category
    """
    for ftr in features:
        df[ftr] = df[ftr].astype("category")
    
    return df


# Our function is so simple. Let's use it.

# In[7]:


data = converter(data,data)


# And now I am going to take a look at the data.

# In[8]:


data.head()


# In appearance there is no change, let's use info method.

# In[9]:


data.info()


# * Our preprocessing is over.

# # Feature Engineering
# 
# In this section I am going to apply pandas' get dummies function. As you guess, in order to do this I am going to define a function. Let's do this.

# In[10]:


x = data.drop("class",axis=1)
y = data["class"]


# But before this I am going to split dataframe x and y axis because I do not want to apply get dummies to y axis.

# In[11]:


def get_dummies(df,features):
    
    for ftr in features:
        
        df = pd.get_dummies(df,ftr)
    
    return df


# * Our function is ready, let's use it.

# In[12]:


x = get_dummies(x,x)


# * Our feature engineering section is over

# # Modeling
# 
# Finally our main stage has come. In this section I am going to train machine learning models. I am going to start this section with train test split. 

# ## Train Test Split
# 
# In this sub-section I am going to split dataframe into two pieces. Train and test. In order to do this I am going to use sklearn library.

# In[13]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)


# And let's look at the lengths of arrays.

# In[14]:


print(("Len of x_train is ",len(x_train)))
print(("Len of x_test is ",len(x_test)))
print(("Len of y_train is ",len(y_train)))
print(("Len of y_test is ",len(y_test)))


# ## Classification Algorithms
# 
# In this section I am going to train classification models however at least for this kernel I am not going to use Grid Search because I think it is so much for this kernel.
# 
# In this section I am going to train these algorithms:
# * Support Vector Machine Classification
# * Decision Tree Classification
# * Naive Bayes Classification
# * Random Forest Classification

# ### Support Vector Machines Classification

# In[15]:


from sklearn.svm import SVC

svc = SVC(random_state=12)
svc.fit(x_train,y_train)

print((svc.score(x_test,y_test)))


# ### Decision Tree Classification

# In[16]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=12)
DTC.fit(x_train,y_train)

print((DTC.score(x_test,y_test)))


# ### Naive Bayes Classification

# In[17]:


from sklearn.naive_bayes import GaussianNB

NBC = GaussianNB()
NBC.fit(x_train,y_train)
print((NBC.score(x_test,y_test)))


# * Naive Bayes score is lower than Decision Tree and Support Vector Machines

# ### Random Forest Classification

# In[18]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=50,random_state=12)
RFC.fit(x_train,y_train)

print((RFC.score(x_test,y_test)))


# # Result | Best Algorithm for Mushroom Classification
# 
# Let's take a look at our scores.
# 
# Support Vector Machines Classifier = %100
# Decision Tree Classifier = %100
# Naive Bayes Classifier = %96
# Random Forest Classifier = %100
# 
# Our scores are great! 
# 
# And at the end of this section we can say: For mushroom classification we can use SVC,DTC or RFC

# # Conclusion 
# 
# Thanks for your attention. If you like this kernel, I would be glad. 
