#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[61]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# ## Data Exploration

# In[62]:


print((train.shape,test.shape))


# In[63]:


train.head()


# In[64]:


test.head()


# In[65]:


train.info()


# In[66]:


test.info()


# In[67]:


train.skew()


# In[68]:


test.skew()


# In[69]:


train.duplicated().sum()


# In[70]:


train.isnull().sum()


# In[71]:


## Categorical v/s Categorical
plt.figure(figsize=(10,6))
plt.title('Count of Passengers Survived')
sns.countplot(train['Pclass'][train['Survived']==1])
plt.show()


# In[72]:


plt.figure(figsize=(10,6))
plt.title('Count of gender of passengers Survived')
sns.countplot(train['Sex'][train['Survived']==1])
plt.show()


# In[73]:


plt.figure(figsize=(10,6))
plt.title('Count of Passengers having Spouses/Siblings Dead')
sns.countplot(train['SibSp'][train['Survived']==0])
plt.show()


# In[74]:


plt.figure(figsize=(10,6))
plt.title('Count of passengers having parents Dead')
sns.countplot(train['Parch'][train['Survived']==0])
plt.show()


# In[75]:


plt.figure(figsize=(10,6))
plt.title('Count of passengers having specific Starting point Dead')
sns.countplot(train['Embarked'][train['Survived']==0])
plt.show()


# In[76]:


# Numeric v/s Categorical
plt.figure(figsize=(12,5))
sns.distplot(train['Age'][train['Survived']==0])
sns.distplot(train['Age'][train['Survived']==1])
plt.legend(['Dead','Survived'])
plt.show()


# In[77]:


plt.figure(figsize=(10,5))
sns.pointplot(x='Embarked', y='Age', hue='Survived', data = train)
plt.show()


# In[78]:


train['Age'].fillna(train['Age'].median(), inplace = True)
test['Age'].fillna(test['Age'].median(), inplace = True)


# In[79]:


train.isnull().sum()


# In[80]:


train.drop(['Name'] , axis = 1, inplace = True)
test.drop(['Name'] , axis = 1, inplace = True)


# In[81]:


test.isnull().sum()


# In[82]:


train.head()


# In[83]:


test.head()


# In[84]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train['Sex'] = labelencoder.fit_transform(train['Sex'])


# In[85]:


train.head(2)


# In[86]:


plt.figure(figsize=(10,5))
sns.countplot(train['Embarked'])


# In[87]:


train['Embarked'].fillna('S', inplace = True)
train['Embarked'] = labelencoder.fit_transform(train['Embarked'])


# In[88]:


train.isnull().sum()


# In[89]:


test.isnull().sum()


# In[90]:


test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace = True)
test.head(50)


# In[91]:


train.drop(['Cabin'], axis = 1, inplace= True)
test.drop(['Cabin'], axis = 1, inplace= True)


# In[92]:


print((test.info(), train.info()))


# In[93]:


train.drop(['Ticket','PassengerId'], axis = 1, inplace= True)
test.drop(['Ticket'], axis = 1, inplace= True)


# In[94]:


train_data = train.drop('Survived', axis = 1)
target = train['Survived']
train_data.shape, target.shape


# In[95]:


test.head()
test['Sex'] = labelencoder.fit_transform(test['Sex'])
test['Embarked'] = labelencoder.fit_transform(test['Embarked'])


# In[96]:


train_data.head(10)


# 0 -> C ; 1 -> Q ; 2 -> S

# ## Modelling

# In[97]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[98]:


train.info()


# ## Cross-Validation(K-Fold)

# In[99]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### KNN

# In[100]:


clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[101]:


round(np.mean(score)*100, 2)


# ### Decision Tree

# In[102]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)


# In[103]:


round(np.mean(score)*100, 2)


# ### Random Forest

# In[104]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)


# In[105]:


round(np.mean(score)*100, 2)


# ### Naive Bayes

# In[106]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)


# In[107]:


round(np.mean(score)*100, 2)


# ### SVM

# In[108]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring= scoring)
print(score)


# In[109]:


round(np.mean(score)*100, 2)


# ## Testing

# In[110]:


clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data, target)
test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[112]:


submission = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : prediction
})
submission.to_csv('submission.csv', index=False)


# In[113]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:




