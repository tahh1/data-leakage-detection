#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Andrey239/ML_LPI_2021/blob/main/seminar04/MLatFIAN2021_seminar04_homework_82.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## Task 1

# (Titanic data again)
# 
# Build a model with `sklearn`'s `LogisticRegression` or `SVC` to get the accuracy of at least 0.81 on the test set. Can you get higher? 0.84?
# 
# Some (optional) suggestions:
# - Add new features (e.g. missing value indicator columns)
# - Fill missing values
# - Encode categorical features (e.g. one-hot encoding)
# - Scale the features (e.g. with standard or robust scaler)
# - Think of other ways of preprocessing the features (e.g. `Fare` $\to$ `log(Fare)`)
# - Try adding polynomial features
# - use `sklearn.model_selection.GridSearchCV` to search for the best hyperparameters (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
# 
# 

# In[2]:


get_ipython().system('wget https://github.com/HSE-LAMBDA/MLatFIAN2021/raw/main/seminar01/train.csv')


# In[3]:


import pandas as pd
data = pd.read_csv("train.csv", index_col='PassengerId')
data.head()


# #### About the data
# Here's some of the columns
# * Name - a string with person's full name
# * Survived - 1 if a person survived the shipwreck, 0 otherwise.
# * Pclass - passenger class. Pclass == 3 is cheap'n'cheerful, Pclass == 1 is for moneybags.
# * Sex - a person's gender
# * Age - age in years, if available
# * SibSp - number of siblings on a ship
# * Parch - number of parents on a ship
# * Fare - ticket cost
# * Embarked - port where the passenger embarked
#  * C = Cherbourg; Q = Queenstown; S = Southampton

# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def feature_selection_and_preprocessing(dataset):
 
  features = dataset[[ "SibSp", "Pclass", "Parch", "Sex"]].copy()
  features["Age"] = np.round(dataset.Age.fillna(dataset.Age.median())/11)
  features["Fare"] = np.round(dataset.Fare/20)
  features["Age*class"]= features.Age * dataset.Pclass
  features["Lonely"] = np.logical_and(dataset.SibSp , 1)
  features["Sex"][features.Sex == "male"] = 1
  features["Sex"][features.Sex == "female"] = 0
  features["Fare per person"] = features.Fare/((dataset.SibSp+1))

  features['Embarked'] = data.Embarked.fillna('unknown')

  return features

print((feature_selection_and_preprocessing(data_train.drop('Survived', axis=1)).head(10)))

model = make_pipeline(
    # <YOUR CODE>
    # E.g.
    make_column_transformer(
        (OneHotEncoder(sparse=False), ['Embarked']),
        remainder='passthrough'
    ),
    LogisticRegression()
)


# Validation code (do not touch)
data = pd.read_csv("train.csv", index_col='PassengerId')
data_train, data_test = train_test_split(data, test_size=200, random_state=42)

model.fit(
    feature_selection_and_preprocessing(
        data_train.drop('Survived', axis=1)
    ),
    data_train['Survived']
)

train_predictions = model.predict(
    feature_selection_and_preprocessing(
        data_train.drop('Survived', axis=1)
    )
)

test_predictions = model.predict(
    feature_selection_and_preprocessing(
        data_test.drop('Survived', axis=1)
    )
)

print(("Train accuracy:", accuracy_score(
    data_train['Survived'],
    train_predictions
)))
print(("Test accuracy:", accuracy_score(
    data_test['Survived'],
    test_predictions
)))


# In[ ]:




