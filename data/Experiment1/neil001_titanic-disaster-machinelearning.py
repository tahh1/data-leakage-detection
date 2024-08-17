#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # <center>Titanic: Disaster_Machine Learning</center>

# ### <u>Context</u>

# The data has been split into two groups:
# 
# * The <b>training set</b> should be used to build your machine learning models. For the training set, the outcome (also known as the “ground truth”) for each passenger. The model will be based on “features” like passengers gender and class. The feature engineering is use to create new features.
# 
# * The <b>test set</b> should be used to see how well the model performs on unseen data. For the test set, the ground truth for each passenger is not available. For each passenger in the test set, the model trained to predict whether or not passengers survived the sinking of the Titanic.

# ### <u>Content</u>

# This dataset contains the below feature column fields:
# 
# * **PassengerId:** Passenger Identity
# * **Survived:** Passenger survived or not
# * **Pclass:** Class of ticket
# * **Name:** Name of passenger
# * **Sex:** Sex of passenger
# * **Age:** Age of passenger
# * **SibSp:** Number of sibling and/or spouse travelling with passenger
# * **Parch:** Number of parent and/or children travelling with passenger
# * **Ticket:** Ticket number
# * **Fare:** Price of ticket
# * **Embarked:** Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# ### <u>Objective</u>

# The goal is to __predict survival__ of passengers travelling in __Titanic__ using various modelling techniques.

# In[2]:


## Importing the required libraries and packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


## Importing the train csv datafile

titanic_train = pd.read_csv('../input/titanic/train.csv')


# In[4]:


## Importing the test csv datafile

titanic_test = pd.read_csv('../input/titanic/test.csv')


# In[5]:


## Description of every column for train data

titanic_train.info()


# In[6]:


## Description of every column for test data

titanic_test.info()


# In[7]:


## Checking top 5 rows of the train dataset

titanic_train.head()


# In[8]:


## Checking top 5 rows of the test dataset

titanic_test.head()


# In[9]:


## Checking for top columns having missing values percentage in train dataset

print(((titanic_train.isnull().sum()/len(titanic_train)*100).round(2).sort_values(ascending = False).head()))


# In[10]:


## Checking for top columns having missing values percentage in test dataset

print(((titanic_test.isnull().sum()/len(titanic_test)*100).round(2).sort_values(ascending = False).head()))


# * From the above train and test dataset we can see that the __Cabin__ column is having high percentage of missing values hence, we will be dropping the __Cabin__ column.
# * We have very minumum percentage of of missing values for __Age__, __Embarked__ & __Fare__ column and we will keep it for further analysis.

# In[11]:


## Dropping the missing value Cabin column from train dataset

titanic_train.drop('Cabin', axis=1, inplace=True)


# In[12]:


## Dropping the missing value Cabin column from test dataset

titanic_test.drop('Cabin', axis=1, inplace=True)


# In[13]:


## Checking for the Age distribution in training dataset

plt.figure(figsize=(10,5))
sns.distplot(titanic_train['Age'] , bins=30)


# In[14]:


## Filling the missing values in Age cloumn with mean values for training dataset

titanic_train['Age'] = (titanic_train['Age'].fillna(titanic_train['Age'].mean())).round()


# In[15]:


## Plotting the count of passenger sex based on Pclass for training set

plt.figure(figsize=(10,5))
sns.countplot(x = 'Sex' , data = titanic_train , hue = 'Pclass' , palette = 'rainbow')
plt.title('Passenger Sex based on Pclass')


# - From the above plot we can see there are more numbers of passengers in __Pclass 3__ and __male__ are more in numbers as compare to __female__.

# In[16]:


## Checking for the Age distribution in test dataset

plt.figure(figsize=(10,5))
sns.distplot(titanic_test['Age'] , bins=30)


# In[17]:


## Filling the missing values in Age cloumn with mean values for test dataset

titanic_test['Age'] = (titanic_test['Age'].fillna(titanic_test['Age'].mean())).round()


# In[18]:


## Plotting the count of passenger sex based on Pclass for test set

plt.figure(figsize=(10,5))
sns.countplot(x = 'Sex' , data = titanic_test , hue = 'Pclass' , palette = 'viridis')
plt.title('Passenger Sex based on Pclass')


# - From the above plot we can see there are more numbers of passengers in __Pclass 3__ and __male__ are more in numbers as compare to __female__.

# In[19]:


## Creating a new feature name Family Size(SibSp+Parch+1) for training set

titanic_train['Family Size'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1


# In[20]:


## Creating a new feature name Family Size(SibSp+Parch+1) for test set

titanic_test['Family Size'] = titanic_test['SibSp'] + titanic_test['Parch'] + 1


# In[21]:


## Dropping columns which doesn't add up to our predictions in training set

titanic_train.drop(['Name','Ticket','SibSp','Parch'] , axis = 1 , inplace = True)


# In[22]:


## Dropping columns which doesn't add up to our predictions in test set

titanic_test.drop(['Name','Ticket','SibSp','Parch'] , axis = 1 , inplace = True)


# In[23]:


## Convert categorical variables for training set into dummy variables (i.e. one-hot encoding)

dummies = pd.get_dummies(titanic_train[['Sex','Embarked']],drop_first=True)

titanic_train_dummy = titanic_train.drop(['Sex','Embarked'] , axis = 1)

titanic_train_dummy = pd.concat([titanic_train_dummy , dummies] , axis = 1)

titanic_train_dummy.info()


# In[24]:


## Convert categorical variables for test set into dummy variables (i.e. one-hot encoding)

dummies = pd.get_dummies(titanic_test[['Sex','Embarked']],drop_first=True)

titanic_test_dummy = titanic_test.drop(['Sex','Embarked'] , axis = 1)

titanic_test_dummy = pd.concat([titanic_test_dummy , dummies] , axis = 1)

titanic_test_dummy['Fare'] = (titanic_test_dummy['Fare'].fillna(titanic_test_dummy['Fare'].mean())).round()

titanic_test_dummy.info()


# In[25]:


## Plotting the count of passenger died and survived

plt.figure(figsize=(10,5))
sns.countplot(titanic_train_dummy['Survived'])
plt.title('Count of passenger died and survived')


# * From the above plot we can say that there are more number of passengers died than survived.

# In[26]:


## Plotting the count of passenger died and survived based on Pclass

plt.figure(figsize=(10,5))
sns.countplot(x = 'Survived' , data = titanic_train_dummy , hue = 'Pclass')
plt.title('Passenger died and survived based on Pclass')


# * From the above plot we can say that the passengers with ticket class __3__ have high mortality count and low survival, and passengers with ticket class __1__ has low mortality count and high survival.

# In[27]:


## Plotting joint relationship between 'Fare' , 'Age' , 'Pclass' & 'Survived' for training dataset

sns.pairplot(titanic_train_dummy[['Fare','Age','Pclass','Survived']] , hue = 'Survived' , height = 4)


# * Observation from above pairplot
#     - More passenger of __Pclass 1__ survived than died
#     - More passenger of __Pclass 3__ died than survived
#     - More passenger of age group __20-40__ died than survived
#     - Most of the passenger paying __less fare__ died

# In[28]:


## Checking correlation between all the features using heatmap for training dataset

plt.figure(figsize=(10,8))
corr = titanic_train_dummy.corr()
sns.heatmap(corr , annot = True , linecolor = 'black' , linewidth = .01)
plt.title('Correlation between features')


# - Observations from above correlation graph
#     - Survived is higly correlated with Fare
#     - Survived is also highly correlated with Pclass
#     
# 

# In[29]:


## Checking correlation between all the features using heatmap for test dataset

plt.figure(figsize=(10,8))
corr = titanic_test_dummy.corr()
sns.heatmap(corr , annot = True , linecolor = 'black' , linewidth = .01 , cmap = 'YlGnBu')
plt.title('Correlation between features')


# ### Model Building

# In[30]:


X_train = titanic_train_dummy.drop(['Survived','PassengerId'] , axis =1)


# In[31]:


y_train = titanic_train_dummy['Survived']


# In[32]:


X_test = titanic_test_dummy.drop("PassengerId", axis=1).copy()


# In[33]:


print(('X_train =' , X_train.shape))
print(('y_train =' , y_train.shape)) 
print(('X_test =' , X_test.shape))


# ### Scaling Data

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[35]:


scaler = StandardScaler()


# In[36]:


X_train = scaler.fit_transform(X_train)


# In[37]:


X_test = scaler.transform(X_test)


# ### Creating a Model (Logistic Regresion)

# In[38]:


from sklearn.linear_model import LogisticRegression


# In[39]:


logreg = LogisticRegression()


# In[40]:


logreg.fit(X_train , y_train)


# ### Model Predictions

# In[41]:


y_pred = logreg.predict(X_test)


# In[42]:


acc_log = (logreg.score(X_train, y_train)*100).round(2)

print(('Accuracy score is:', acc_log))


# ### Creating a Model (Support Vector Machine)

# In[43]:


from sklearn.svm import SVC


# In[44]:


svc = SVC()


# In[45]:


svc.fit(X_train , y_train)


# ### Model Predictions

# In[46]:


y_pred = svc.predict(X_test)


# In[47]:


acc_svm = (svc.score(X_train, y_train)*100).round(2)

print(('Accuracy score is:', acc_svm))


# ### Creating a Model (K - Nearest Neighbours)

# In[48]:


from sklearn.neighbors import KNeighborsClassifier


# In[49]:


knn = KNeighborsClassifier(n_neighbors = 5)


# In[50]:


knn.fit(X_train , y_train)


# ### Model Predictions

# In[51]:


y_pred = knn.predict(X_test)


# In[52]:


acc_knn = (knn.score(X_train, y_train)*100).round(2)

print(('Accuracy score is:', acc_knn))


# ### Creating a Model (Naive Bayes Classifier)

# In[53]:


from sklearn.naive_bayes import GaussianNB


# In[54]:


gaussian = GaussianNB()


# In[55]:


gaussian.fit(X_train , y_train)


# ### Model Predictions

# In[56]:


y_pred = gaussian.predict(X_test)


# In[57]:


acc_gaussian = (gaussian.score(X_train, y_train)*100).round(2)

print(('Accuracy score is:', acc_gaussian))


# ### Creating a Model (Random Forest Classifier)

# In[58]:


from sklearn.ensemble import RandomForestClassifier


# In[59]:


random_forest = RandomForestClassifier(n_estimators = 100)


# In[60]:


random_forest.fit(X_train , y_train)


# ### Model Predictions

# In[61]:


y_pred = random_forest.predict(X_test)


# In[62]:


acc_random_forest = (random_forest.score(X_train, y_train)*100).round(2)

print(('Accuracy score is:', acc_random_forest))


# ### Model Scores 

# In[63]:


model = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machine','K - Nearest Neighbours',
              'Naive Bayes Classifier','Random Forest Classifier'],
    'Score': [acc_log, acc_svm, acc_knn, acc_gaussian, acc_random_forest]})

model.sort_values(by = 'Score' , ascending = False)


# * From the above models __Logistic Regression__, __Support Vector Machine__, __K-Nearest Neighbours__, __Naive Bayes Classifier__ and __Random Forest Classifier__ it can be conclude that __Random Forest Classifier__ model out performs the best with an __Accuracy Score of 97.87__.

# In[64]:


final_pred = pd.DataFrame({
        "PassengerId": titanic_test_dummy['PassengerId'],
        "Survived": y_pred
    })


# In[65]:


## Predicting the survival and writing to CSV

final_pred.to_csv('gender_submission.csv', index = False)


# In[66]:


titanic_survival = pd.read_csv('gender_submission.csv')


# In[67]:


titanic_survival.info()


# In[68]:


print((titanic_survival['Survived'].value_counts()))

