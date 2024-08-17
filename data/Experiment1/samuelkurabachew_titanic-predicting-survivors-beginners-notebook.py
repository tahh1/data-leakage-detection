#!/usr/bin/env python
# coding: utf-8

# # Predicting Titanic Survivors - Beginners Guide - Part 1
# 
# This will be the first of kernels that will come in the future with minimal and basic analysis code.
# 
# I suggest every beginner should start from this kernel as the others continuation of this (making better predictions).
# 
# This kernel is not ideal but does the job. In the future we will do more EDA, feature engineering and eventually get better accuracy.
# 
# Here is the link to part two. [Titanic - Predicting Survivors - Part 2](https://www.kaggle.com/samuelkurabachew/titanic-predicting-survivors-part-2/output?scriptVersionId=29935993)
# 

# In[1]:


# Import some important libraries

import numpy as np 
import pandas as pd  


# In[2]:


train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")


# In[3]:


print(("Train data size: ", train_data.shape))
print(("Test data size: ", test_data.shape))


# In[4]:


train_data.head()


# In[5]:


test_data.head()


# In[6]:


train_data.describe()


# In[7]:


print('\n****************** Train info ****************\n')
train_data.info()

print('\n****************** Test info ****************\n')
test_data.info()


# From the above information we have a total of 891 entries in train and 418 in test data with some columns missing values in both datasets. To make it clear let see below.

# In[8]:


train_data_na = (train_data.isnull().sum() / len(train_data)) * 100
train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Train data Missing Ratio' :train_data_na})
missing_data


# Now we have to take care of this missing values. 
# We can use different stratagies for this.
# Let's also drop **Cabin** columns since more that a half of the data is missing.

# In[9]:


train_data.drop("Cabin", axis=1, inplace=True)


# In[10]:


train_data["Age"].fillna(train_data["Age"].median(), inplace = True)


# Since Embarked columns has object data type, we can use the most common value among it to fill the missing values.

# In[11]:


train_data["Embarked"].value_counts(ascending=False)


# In[12]:


train_data["Embarked"].fillna("S", inplace=True)


# Now lets check if we have any missing values in the training data again.

# In[13]:


train_data.info()


# Well, it seems data we have taken care of all the missing values in the training data. Now, let's move into the test data.

# In[14]:


test_data_na = (test_data.isnull().sum() / len(test_data)) * 100
test_data_na = test_data_na.drop(test_data_na[test_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Test data Missing Ratio' :test_data_na})
missing_data


# In[15]:


test_data.drop("Cabin", axis=1, inplace=True)
test_data["Age"].fillna(test_data["Age"].median(), inplace = True)
test_data["Fare"].fillna(test_data["Fare"].median(), inplace = True)


# In[16]:


test_data.info()


# We have some categorical data, but our models need numerical data. Therefore we need to change them to numerical data.
# 
# Before we do this, we might have to join our train test data in order to have the same number of columns after encoding.

# In[17]:


n_train = train_data.shape[0]
n_test = test_data.shape[0]

X_train = train_data.drop("Survived", axis=1)
y_train = pd.DataFrame(train_data["Survived"], columns=["Survived"])

all_data = pd.concat((X_train, test_data))


# In[18]:


all_data.tail()


# In[19]:


# Drop Name column since it won't have any effect on our model (This is only for now. In the future there is important
# info we can extract).

all_data.drop('Name', axis=1, inplace=True)
all_data = pd.get_dummies(all_data)


# In[20]:


all_data.shape


# In[21]:


X_train = all_data[:n_train]
test_data = all_data[n_train:]


# # Let's predict the survivors 

# We can use different models and compare thier accuracy to choose the best.
# 
# But before that let's make a validation data to prevent our model from overfitting.

# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[23]:


print((X_train.shape))
print((X_test.shape))
print((y_train.shape))
print((y_test.shape))


# In[24]:


# Import models.
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[25]:


# Import accuracy metrics since it's how our model is evaluated
from sklearn.metrics import accuracy_score


# ## SVC model

# In[26]:


svc_clf = SVC() 
svc_clf.fit(X_train, y_train)
pred_svc = svc_clf.predict(X_test)
acc_svc = accuracy_score(y_test, pred_svc)


# ## Random Forest Classifier

# In[27]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
pred_rf = rf_clf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)


# ## KNN

# In[28]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
pred_knn = knn_clf.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)


# ## GaussianNB

# In[29]:


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
pred_gnb = gnb_clf.predict(X_test)
acc_gnb = accuracy_score(y_test, pred_gnb)


# ## DecisionTree

# In[30]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
pred_dt = dt_clf.predict(X_test)
acc_dt = accuracy_score(y_test, pred_dt)


# ## XGBoost

# In[31]:


from xgboost import XGBClassifier

xg_clf = XGBClassifier()
xg_clf.fit(X_train, y_train)
pred_xg = xg_clf.predict(X_test)
acc_xg = accuracy_score(y_test, pred_xg)

print(acc_xg)


# Let's evaluate our models.

# In[32]:


model_performance = pd.DataFrame({
    "Model": ["SVC", 
              "Random Forest", 
              "K Nearest Neighbors", 
              "Gaussian Naive Bayes",  
              "Decision Tree", 
              "XGBClassifier"],
    "Accuracy": [acc_svc, 
                 acc_rf, 
                 acc_knn, 
                 acc_gnb, 
                 acc_dt, 
                 acc_xg]
})

model_performance.sort_values(by="Accuracy", ascending=False)


# As seen from above we can get pretty much good accuracy on Random Forest, Decision tree and XGB.

# # Submission

# In[33]:


submission_predictions = rf_clf.predict(test_data)

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": submission_predictions
    })

submission.to_csv("survivors.csv", index=False)
print((submission.shape))


# In[ ]:




