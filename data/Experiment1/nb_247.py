#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from random import randint

from sklearn.preprocessing import LabelEncoder, StandardScaler
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import Support Vector Machine
from sklearn.svm import SVC
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
# Import metrics
from sklearn.metrics import confusion_matrix, precision_score, \
    recall_score, roc_curve, roc_auc_score, f1_score
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV
# Import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


os.makedirs("../data", exist_ok=True)

get_ipython().system('wget -qO ../data/Churn.csv "https://assets.datacamp.com/production/repositories/1764/datasets/79c5446a4a753e728e32b4a67138344847b8f131/Churn.csv"')


# ## Exploratory data analysis

# In[3]:


# Importing data using pandas
telco = pd.read_csv('../data/Churn.csv')
telco.head()


# In[4]:


# Data type
telco.info()


# In[5]:


# check any NA value
telco[telco.isna().any(axis=1)]


# In[6]:


print((telco['Churn'].value_counts()))
pd01 = telco['Churn'].value_counts(normalize=True)*100
pd01.plot.bar()
plt.title('% Churner')
plt.xlabel('Churn')
plt.ylabel('% churner')
plt.show()


# In[7]:


# Group telco by 'Churn' and compute the mean
telco.groupby(['Churn']).mean()


# In[8]:


# Group telco by 'Churn' and compute the std
telco.groupby(['Churn']).std()


# In[9]:


# Count the number of churners and non-churners by State
print((telco.groupby('State')['Churn'].value_counts()))
pd01 = pd.DataFrame(telco.groupby('State')['Churn'].value_counts())\
    .unstack(level=1)
pd01.plot.bar(figsize=(20, 10))
plt.title('# Churner by State')
plt.xlabel('State')
plt.ylabel('# Churner')
plt.show()


# In[10]:


# Visualizing the distribution of account lengths
# Important to understand how variables are distributed
plt.figure(figsize=(10, 5))
sns.histplot(telco['Account_Length'], kde=True, stat='density',
    linewidth=0)
plt.show()


# In[11]:


# Exploring distribution of variables
plt.figure(figsize=(10, 10))

plt.subplot(4, 1, 1)
sns.histplot(telco['Day_Mins'], kde=True, stat='density',
    linewidth=0)

plt.subplot(4, 1, 2)
sns.histplot(telco['Eve_Mins'], kde=True, stat='density',
    linewidth=0)

plt.subplot(4, 1, 3)
sns.histplot(telco['Night_Mins'], kde=True, stat='density',
    linewidth=0)

plt.subplot(4, 1, 4)
sns.histplot(telco['Intl_Mins'], kde=True, stat='density',
    linewidth=0)

plt.tight_layout()
plt.show()


# In[12]:


# Differences in account length
plt.figure(figsize=(10, 5))
sns.boxplot(data=telco, x='Churn', y='Account_Length', 
    sym="")
plt.show()


# In[13]:


# Adding a third variable
plt.figure(figsize=(10, 5))
sns.boxplot(data=telco, x='Churn', y='Account_Length', 
    hue='Intl_Plan')
plt.show()


# In[14]:


# Create the box plot
plt.figure(figsize=(10, 5))
sns.boxplot(data=telco, x='Churn', y='CustServ_Calls',
    sym = "", hue="Intl_Plan")
plt.show()


# ## Feature engineering

# In[15]:


# Dropping unnecessary features
telco = telco.drop(['Area_Code', 'Phone'], axis=1)
telco.columns


# In[16]:


# Compute correlated features
telco.corr()


# In[17]:


# Dropping correlated features
telco = telco.drop(['Day_Charge', 'Eve_Charge', 
    'Night_Charge', 'Intl_Charge'], axis=1)
telco.columns


# In[18]:


# Feature Engineering
# Create the new feature
telco['Avg_Day_Calls'] = np.where(telco['Day_Calls']==0, 0, telco['Day_Mins'] / telco['Day_Calls'])
telco['Avg_Eve_Calls'] = np.where(telco['Eve_Calls']==0, 0, telco['Eve_Mins'] / telco['Eve_Calls'])
telco['Avg_Night_Calls'] = np.where(telco['Night_Calls']==0, 0, telco['Night_Mins'] / telco['Night_Calls'])
telco['Avg_Intl_Calls'] = np.where(telco['Intl_Calls']==0, 0, telco['Intl_Mins'] / telco['Intl_Calls'])
# Print the summary statistic
telco[['Avg_Day_Calls', 'Avg_Eve_Calls', 'Avg_Night_Calls', 
    'Avg_Intl_Calls']].describe().round(2)


# In[19]:


# Encoding binary features
telco['Intl_Plan'] = LabelEncoder().fit_transform(telco['Intl_Plan'])
telco['Vmail_Plan'] = LabelEncoder().fit_transform(telco['Vmail_Plan'])
# Replace 'no' with 0 and 'yes' with 1 in 'Churn'
telco['Churn'] = telco['Churn'].replace({'no':0, 'yes':1})

# Print the results to verify
telco[['Intl_Plan', 'Vmail_Plan', 'Churn']].describe().round(2)


# In[20]:


# Perform one hot encoding on 'State' 
tmp = pd.get_dummies(telco['State'], prefix='State')
telco = pd.concat([telco, tmp], axis=1)
telco.drop(['State'], axis=1, inplace=True)
# Print the summary statistic
telco.describe().round(2)


# In[21]:


# Scale telco using StandardScaler
telco[['Account_Length', 'Vmail_Message', 'Day_Mins', 'Eve_Mins', 
    'Night_Mins', 'Intl_Mins', 'CustServ_Calls', 'Day_Calls', 
    'Eve_Calls', 'Night_Calls', 'Intl_Calls', 'Avg_Day_Calls', 
    'Avg_Eve_Calls', 'Avg_Night_Calls', 'Avg_Intl_Calls']] = \
StandardScaler().fit_transform(telco[[
    'Account_Length', 'Vmail_Message', 'Day_Mins', 'Eve_Mins', 
    'Night_Mins', 'Intl_Mins', 'CustServ_Calls', 'Day_Calls', 
    'Eve_Calls', 'Night_Calls', 'Intl_Calls', 'Avg_Day_Calls', 
    'Avg_Eve_Calls', 'Avg_Night_Calls', 'Avg_Intl_Calls']])
telco.reset_index(inplace=True, drop=True)
# Print summary statistics
telco.describe().round(2)


# ## Churn prediction

# In[22]:


# Create feature variable
X = telco.drop('Churn', axis=1)
# Create target variable
y = telco['Churn']
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print((X_train.shape))
print((len(y_test)))


# In[23]:


# Instantiate the classifier
svc = SVC()
# Fit the classifier
svc.fit(X_train, y_train)
# Predict the label of new_customer
prediction = svc.predict(X_test)
# Computing accuracy
svc.score(X_test, y_test)


# In[24]:


# Instantiate the classifier
clf = LogisticRegression()
# Fit the classifier
clf.fit(X_train, y_train)
# Computing accuracy
print((clf.score(X_test, y_test)))
# Generate probability
y_pred_prob = clf.predict_proba(X_test)[:, 1]
# Calculate the roc metrics
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot the ROC curve
plt.plot(fpr, tpr)
# Add labels and diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0, 1], [0, 1], 'k--')
plt.show()
# Print the AUC
print((roc_auc_score(y_test, y_pred_prob)))


# In[25]:


# Instantiate the classifier
clf = DecisionTreeClassifier()
# Fit the classifier
clf.fit(X_train, y_train)
# Computing accuracy
clf.score(X_test, y_test)


# In[26]:


# Instantiate the classifier
clf = RandomForestClassifier()
# Fit the classifier
clf.fit(X_train, y_train)
# Predict the label of X_test
y_pred = clf.predict(X_test)
# Computing accuracy
clf.score(X_test, y_test)


# ## Model metrics

# In[27]:


# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[32]:


# Create feature variable
X = telco.drop('Churn', axis=1)
# Create target variable
y = telco['Churn']
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
# Instantiate the classifier
clf = RandomForestClassifier()
# Fit to the training data
clf.fit(X_train, y_train)
# Predict the labels of the test set
y_pred = clf.predict(X_test)
# Print confusion matrix
print(("Confusion matrix:\n", confusion_matrix(y_test, y_pred)))
# Print the precision
print(("Precision:", precision_score(y_test, y_pred)))
# Print the recall
print(("Recall:", recall_score(y_test, y_pred)))
# Generate the probabilities
y_pred_prob = clf.predict_proba(X_test)[:, 1]
# Calculate the roc metrics
fpr, tpr, thresholds = roc_curve(
    y_test, y_pred_prob)
# Plot the ROC curve
plt.plot(fpr, tpr)
# Add labels and diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0, 1], [0, 1], "k--")
plt.show()
# Print the AUC
print(("AUC:", roc_auc_score(y_test, y_pred_prob)))
# Print the F1 score
print(("F1 score:", f1_score(y_test, y_pred)))


# ## Hyperparameter tuning

# In[35]:


# Create the hyperparameter grid
param_grid = {'max_features': ['auto', 'sqrt', 'log2']}
# Call GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
# Fit the model
grid_search.fit(X, y)
# Print the optimal parameters
print((grid_search.best_params_))
print((grid_search.best_score_))


# In[37]:


# Create the hyperparameter grid
param_grid = {"max_depth": [3, None],
    "max_features": [1, 3, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]}
# Call GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
# Fit the model
grid_search.fit(X, y)
# Print the optimal parameters
print((grid_search.best_params_))


# In[46]:


# Instantiate the classifier
clf = RandomForestClassifier()
# Create the hyperparameter grid
param_dist = {"max_depth": [3, None],
              "max_features": [randint(1, 11)],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# Call RandomizedSearchCV
random_search = RandomizedSearchCV(clf, param_dist, n_iter=8)
# Fit the model
random_search.fit(X, y)
# Print best parameters
print((random_search.best_params_))


# ## Feature importances

# In[50]:


# Instantiate the classifier
clf = RandomForestClassifier(max_features=7, max_depth=None,
    criterion='gini', bootstrap=False)
# Fit the model
clf.fit(X, y)
# Calculate feature importances
importances = clf.feature_importances_
# Sort importances
sorted_index = np.argsort(importances)
# Create labels
labels = X.columns[sorted_index]
# Create plot
plt.figure(figsize=(10, 20))
plt.barh(list(range(X.shape[1])), importances[sorted_index], 
    tick_label=labels)
plt.show()


# In[52]:


# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)
# Instantiate the classifier
clf = RandomForestClassifier()
# Fit to the data
clf.fit(X_train, y_train)
# Print the accuracy
print(("Accuracy:", clf.score(X_test, y_test)))
# Predict the labels of the test set
y_pred = clf.predict(X_test)
# Print the F1 score
print(("F1 score:", f1_score(y_test, y_pred)))

