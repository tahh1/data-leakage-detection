#!/usr/bin/env python
# coding: utf-8

# **Importing libraries**

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

#machine learning models to implement 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# **Data preprocessing**

# In[2]:


traindf =  pd.read_csv('../input/titanic/train.csv')
traindf


# In[3]:


testdf = pd.read_csv('../input/titanic/test.csv')
testdf


# In[4]:


combine = [traindf,testdf]


# In[5]:


traindf.describe


# **Data visualization**

# In[6]:


sns.countplot('Survived', data = traindf)


# In[7]:


plt.figure(figsize=(10,8))
sns.heatmap(traindf.corr(), cmap='coolwarm',annot=True)


# In[8]:


sns.countplot('Sex', hue = 'Survived', data = traindf)


# In[9]:


sns.countplot('Pclass', hue='Survived', data = traindf)


# In[10]:


sns.countplot('Survived', hue='Embarked', data =traindf)


# In[11]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(traindf, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


#  **Feature engineering**

# In[12]:


#dropping features 
print(('Before', traindf.shape, testdf.shape, combine[0].shape, combine[1].shape))

traindf.drop(['Ticket','Cabin'], axis=1, inplace=True)
testdf.drop(['Ticket','Cabin'], axis=1 , inplace=True)
combine =[traindf,testdf]

print(('After', traindf.shape, testdf.shape, combine[0].shape, combine[1].shape ))


# In[13]:


#CREATING A NEW FEATURE FROM EXISITING 
for dataset in combine : 
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(traindf['Title'],traindf['Sex'])


# *Replacing titles with common name*

# In[14]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    
traindf[['Title','Survived']].groupby(['Title'], as_index=False).mean()


# *Convert categorical variables into ordinal*

# In[15]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
traindf.head()


# In[16]:


traindf = traindf.drop(['Name', 'PassengerId'], axis=1)
testdf = testdf.drop(['Name'], axis=1)
combine = [traindf, testdf]
traindf.shape, testdf.shape


# In[17]:


#encoding categorical data 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
embark_mapping =  {"Q":1, "S":2, "C":3}
for dataset in combine:
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping)
    dataset['Embarked'] = dataset['Embarked'].fillna(0)
    
traindf.head()


# **Handling Missing values**

# In[18]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
traindf[['Age']] = imputer.fit_transform(traindf[['Age']])
testdf[['Age']] = imputer.fit_transform(testdf[['Age']])


# In[19]:


for dataset in combine:
    dataset.columns[dataset.isnull().any()]
    nanfare =  pd.isnull(dataset["Fare"])
dataset[nanfare]


# In[20]:


for dataset in combine:
    dataset[['Fare']] = imputer.fit_transform(dataset[['Fare']])


# In[21]:


dataset.columns[dataset.isnull().any()]


# In[22]:


traindf.head()


# In[23]:


testdf.head()


# **MODEL TRAINING**

# In[24]:


X_train = traindf.drop("Survived", axis=1)
y_train = traindf["Survived"]
X_test = testdf.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape


# In[25]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[26]:


#LOGISTIC REGRESSION 
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred =  lr.predict(X_test)
acc_lr = round(lr.score(X_train,y_train)*100, 2)
acc_lr


# In[27]:


coeff_df = pd.DataFrame(traindf.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(lr.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# *OBSERVATIONS::*
# * Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# * Inversely as Pclass increases, probability of Survived=1 decreases the most.
# * This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# * So is Title as second highest positive correlation.

# In[28]:


#GAUSSIAN NAIVE BAYES 
gauss = GaussianNB()
gauss.fit(X_train, y_train)
y_pred =  gauss.predict(X_test)
acc_gauss = round(gauss.score(X_train,y_train)*100, 2)
acc_gauss


# In[29]:


#PERCEPTRON
per = Perceptron()
per.fit(X_train,y_train)
y_pred = per.predict(X_test)
acc_per = round(per.score(X_train,y_train)*100, 2)
acc_per


# In[30]:


#STOCHASTIC GRADIENT DESCENT
sgd = SGDClassifier()
sgd.fit(X_train,y_train)
y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train,y_train)*100, 2)
acc_sgd


# In[31]:


#K-Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# In[32]:


#SUPPORT VECTOR MACHINES 
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc


# In[33]:


#RANDOM FOREST
ran =  RandomForestClassifier(n_estimators = 100)
ran.fit(X_train,y_train)
y_pred = ran.predict(X_test)
acc_ran = round(ran.score(X_train,y_train)*100, 2)
acc_ran


# **MODEL EVALUATION**

# In[34]:


models =  pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'GaussianNB',
              'Perceptron', 'Random Forest', 'K-Neighbors', 'SGD', 'Support Vector Machines'],
    'Score': [acc_lr, acc_knn, acc_gauss,
             acc_per, acc_ran, acc_knn, acc_sgd, acc_svc]})
models.sort_values(by='Score', ascending=False)


# In[35]:


submission_df = pd.DataFrame
submission_df = pd.DataFrame({
    "PassengerId" : testdf["PassengerId"],
    "Survived" : y_pred
})

submission_df.PassengerId = submission_df.PassengerId.astype(int)
submission_df.Survived = submission_df.Survived.astype(int)
submission_df.to_csv('submission.csv', header=True, index=False)
submission_df.head(10)

