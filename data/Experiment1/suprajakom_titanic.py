#!/usr/bin/env python
# coding: utf-8

# # Import required libraries

# > This is a beginner version of titanic disaster ML competition. I have referred to several notebooks on kaggle that helped beginners like me to work on this problem.

# In[96]:


# Data Analysis libraries

import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


# # Read and explore your input data

# In[97]:


# Read the data into dataframes

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# In[98]:


# Combine train and test data

combine = [train, test]
train.head()


# In[99]:


# Glance the data
train.describe(include='all')


# In[100]:


# List the columns

train.columns.values


# In[101]:


# Data info
train.info()
print(('+-'*20))
test.info()


# ## Some observations and Predictions

#  - Numerical features - Passenger ID, Age, Fare, 
#  - Categorical features - Survived, Sex, Embarked, Passenger Class
#  - Integer features - Ticket, Cabin
#  - Few features that aren't self-explanatory
#      - SibSp - Sibling or spouse
#      - Parch - Parent or child
# 

# In[102]:


# Check for nulls
print((pd.isnull(train).sum()))


# - Total number of passengers is 891.
# - Age is missing for ~19.8% of the records. We may want to fill the gaps as age is an important feature.
# - Cabin feature missing for ~ 77% of records, this might be because only specific passengers have cabins allocated to them.
# - Embarked feature is missing for a small % which is okay.
# - Children may have greater chances of survival
# - Women may have greater chances of survival
# - Passengers in 1st class can have greater chances of survival

# # Data Visualization

# ## Sex Feature

# In[103]:


# Bar plot - Sex vs Survived
sns.barplot(x="Sex", y="Survived", data= train)

# Printing percentages
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# As predicted, chances of survival for females is greater than males. We need to have this feature in our predictions.

# ## Passenger Class Feature

# In[104]:


# Barplot PClass vs Survived
sns.barplot(x="Pclass", y= "Survived", data=train)

#Printing percentages
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# As predicted, higher the class, higher the survival rate. We need to have this feature in our predictions.

# ## Sibsp Feature

# In[105]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

# Printing Percentages
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by="Survived", ascending=False)


# ## Parch feature 

# In[106]:


#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()

train[["Parch","Survived"]].groupby(['Parch'],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[107]:


print((train.columns.values))


# ## Title Feature

# In[108]:


# We want to extract the designation of names from the combined data set.
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[109]:


# Replace the titles with groups
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



# In[110]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).sum()    


# In[111]:


#map each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# ## Age Feature

# In[112]:


print((pd.isnull(train).sum()))


# In[113]:


train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

# Plot age vs survival

sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# We can conclude that "Babies" have more chance of survival.

# ## Cabin Feature

# Cabin feature is available only less than 30% of the records. This might be in relation to the passenger class. 

# In[114]:


#draw a bar plot for Parch vs. survival

train["Cabin_value"]= pd.notnull(train.loc[:,'Cabin']).astype(int)
print((train.columns.values))                             

test["Cabin_value"]= pd.notnull(test.loc[:,'Cabin']).astype(int)
print((test.columns.values)) 


# In[115]:


pd.crosstab(train['Cabin_value'], train['Pclass'])


# We can conclude that passengers with higher class have cabin values allocated and passengers with lower class have mostly no cabins allocated. Therefore, we can consider a non-blank cabin value tied to more survival rate.

# In[116]:


pd.crosstab(train['Cabin_value'], train['Survived'])


# In[117]:


sns.barplot(x="Cabin_value", y="Survived", data=train)
plt.show()

train[["Cabin_value","Survived"]].groupby(['Cabin_value'],as_index=False).mean().sort_values(by="Survived",ascending=False)


# we can see that surival rate is higher with non-blank cabin field.

# # Data Cleansing

# In[118]:


# See the test data
test.describe(include = "all")


# ## Data Cleansing and Feature Dropping

# Before we drop the features, lets clean the data y filling nulls with data. This has to be done for
# - Age
# - Embarked 

# We can try and use titles as a factor for age filling
#  - Mr. title for "Adult"
#  - Miss. title for "Youth"
#  - Mrs. title for "Adult"
#  - Master title for "Baby"
#  - Royal title for "Adult"
#  - Rare title for "Adult"
# 

# In[119]:


# Check for nulls
print((pd.isnull(train).sum()))


# In[120]:


# Map age to a numerical value and drop the age feature
combine = [train, test]

title_mapping = {"Unknown": 0, "Baby": 1, "Child": 2, "Teenager": 3, "Student": 4, "Young Adult": 5, "Adult": 6, "Senior":7}
for dataset in combine:
    dataset['AgeGroup'] = dataset['AgeGroup'].map(title_mapping)
    dataset['AgeGroup'] = dataset['AgeGroup'].fillna(0)

train.head()


# In[121]:


# Map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# ## New feature for family size

# In[122]:


combine = [train, test]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[123]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# ## Missing Embarked feature

# In[124]:


# fiilling with most occuring value through mode

freq_port = train.Embarked.dropna().mode()[0]
freq_port


# In[125]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[126]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()


# In[127]:


print((train.columns.values))


#  The following features can be dropped
#  - Passenger ID
#  - Pasenger Name( as we extracted the designations)
#  - Age (as we took age group)
#  - Ticket
#  - Cabin (as we took cabin value)

# In[128]:


train = train.drop(["PassengerId","Name","Age","Ticket","SibSp","Parch","Fare","Cabin"], axis=1)


# In[129]:


test = test.drop(["Name","Age","Ticket","SibSp","Parch","Fare","Cabin"], axis=1)


# In[130]:


print((train.columns.values))
print((test.columns.values))


# In[131]:


print((train.info()))
print(('+-'*20))
print((test.info()))


# In[132]:


test.head()


# ## Modelling

# ### Logistic Regression

# In[133]:


X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape


# In[134]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log


# ### Support Vector Machines

# In[135]:


# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_linear_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_linear_svc


# ### KNeighbours

# In[136]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# ### Gaussian NB

# In[137]:


from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian


# ### Decision Tree

# In[138]:


from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# ### Random Forest

# In[139]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[140]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[141]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })


# In[142]:


submission.to_csv('titanic.csv', index=False)


# In[ ]:




