#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# DATA
dataset= pd.read_csv("../input/train.csv")


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
seaborn.set() 


# In[4]:


### Analysing Data: Survived by Class ###
surv_class = dataset[dataset['Survived']==1]['Pclass'].value_counts()
dead_class = dataset[dataset['Survived']==0]['Pclass'].value_counts()
df_class = pd.DataFrame([surv_class, dead_class])
df_class.index=['Survived', 'Died']
df_class.plot(kind='bar',stacked=True, title="Survived/Died by Class", figsize=(5,3))


# In[5]:


Class1_surv = df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100
Class2_surv = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100
Class3_surv = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100
print(("Percentage of Survivors of 1º Class:", round(Class1_surv), "%"))
print(("Percentage of Survivors of 2º Class:" ,round(Class2_surv), "%"))
print(("Percentage of Surviviros of 3º Class:" ,round(Class3_surv), "%"))


# In[6]:


# Survivors by Port
surv_port = dataset[dataset['Survived']==1]["Embarked"].value_counts()
dead_port = dataset[dataset['Survived']==0]["Embarked"].value_counts()
df_embarked = pd.DataFrame([surv_port, dead_port])
df_embarked.index = ['Survived', 'Dead']
df_embarked.plot(kind = 'bar', stacked = True, title="Survived/Died by Port")


# In[7]:


# Categorizing Fare feature:

def FareFunc(data):
    data.loc[data['Fare'].isnull(), 'Fare'] = 7            
    data['FareCat'] = 0
    data.loc[data['Fare'] < 8, 'FareCat'] = 0
    data.loc[(data['Fare'] >= 8 ) & (data['Fare'] < 16),'FareCat' ] = 1
    data.loc[(data['Fare'] >= 16) & (data['Fare'] < 30),'FareCat' ] = 2
    data.loc[(data['Fare'] >= 30) & (data['Fare'] < 45),'FareCat' ] = 3
    data.loc[(data['Fare'] >= 45) & (data['Fare'] < 80),'FareCat' ] = 4
    data.loc[(data['Fare'] >= 80) & (data['Fare'] < 160),'FareCat' ] = 5
    data.loc[(data['Fare'] >= 160) & (data['Fare'] < 270),'FareCat' ] = 6
    data.loc[(data['Fare'] >= 270), 'FareCat'] = 7
FareFunc(dataset)


# In[8]:


surv_fare = dataset[dataset['Survived']==1]["FareCat"].value_counts()
dead_fare = dataset[dataset['Survived']==0]["FareCat"].value_counts()
df_fare = pd.DataFrame([surv_fare, dead_fare])
df_fare.index = ['Survived', 'Dead']
df_fare.plot(kind = 'bar', stacked = False, title="Survived/Died by Fare")


# In[9]:


# Creating FamlSize Feature:
def FamlSize(data):
    data['FamlSize'] = 0
    data['FamlSize'] = data['SibSp'] + data['Parch'] + 1
FamlSize(dataset)


# In[10]:


dataset.head()


# In[11]:


#Verifying Port's
dataset.Embarked.value_counts(normalize = True).plot(kind= "bar")
plt.title("Embarked")


# In[12]:


#Filling NA:
row_index = dataset.Embarked.isnull()
dataset.loc[row_index,'Embarked']='S' 


# In[13]:


#Categorizing Port as int:
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
dataset.Embarked=labelEncoder_X.fit_transform(dataset.Embarked)


# In[14]:


# Filling NA Age:
print(('Number of null values in Age:', sum(dataset.Age.isnull())))


# In[15]:


got = dataset.Name.str.split(',').str[1]
dataset.iloc[:,3]=pd.DataFrame(got).Name.str.split('\s+').str[1]
ax = plt.subplot()
ax.set_ylabel('Average age')
dataset.groupby('Name').mean()['Age'].plot(kind = 'bar', figsize=(13,8), ax = ax)


# In[16]:


# Filling Age by title
title_mean_age=[]
title_mean_age.append(list(set(dataset.Name))) 
title_mean_age.append(dataset.groupby('Name').Age.mean())
title_mean_age

n_traning= dataset.shape[0]   
n_titles= len(title_mean_age[1])
for i in range(0, n_traning):
    if np.isnan(dataset.Age[i])==True:
        for j in range(0, n_titles):
            if dataset.Name[i] == title_mean_age[0][j]:
                dataset.Age[i] = title_mean_age[1][j]


# In[17]:


dataset["Title"] = dataset['Name']
dataset=dataset.drop(['Name'], axis=1)


# In[18]:


dataset.head()


# In[19]:


#Categorizing Title as int:
dataset.Title=labelEncoder_X.fit_transform(dataset.Title)


# In[20]:


# Categorizing Age:
# Lets derive AgeGroup feature from age
def AgeCat(data):
    data['AgeCat'] = 0
    data.loc[(data['Age'] <= 5), 'AgeCat'] = 0
    data.loc[(data['Age'] <= 12) & (data['Age'] > 5), 'AgeCat'] = 1
    data.loc[(data['Age'] <= 18) & (data['Age'] > 12), 'AgeCat'] = 2
    data.loc[(data['Age'] <= 22) & (data['Age'] > 18), 'AgeCat'] = 3
    data.loc[(data['Age'] <= 32) & (data['Age'] > 22), 'AgeCat'] = 4
    data.loc[(data['Age'] <= 45) & (data['Age'] > 32), 'AgeCat'] = 5
    data.loc[(data['Age'] <= 60) & (data['Age'] > 45), 'AgeCat'] = 6
    data.loc[(data['Age'] <= 70) & (data['Age'] > 60), 'AgeCat'] = 7
    data.loc[(data['Age'] > 70), 'AgeCat'] = 8
AgeCat(dataset)


# In[21]:


dataset.head()


# In[22]:


# Categorizing Sex as int:
dataset.Sex=labelEncoder_X.fit_transform(dataset.Sex)


# In[23]:


from sklearn.preprocessing import StandardScaler
target = dataset['Survived'].values
select_features = ['Pclass', 'Age','AgeCat','SibSp', 'Parch', 'Fare', 
                   'Embarked', 'Title',
                   'FareCat', 'FamlSize','Sex']
scaler = StandardScaler()
dfScaled = scaler.fit_transform(dataset[select_features])
train = dfScaled[0:891].copy()
test = dfScaled[891:].copy()

# Checking best features
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, len(select_features))
selector.fit(train, target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]

print('Features importance:')
for i in range(len(scores)):
    print(('%.2f %s' % (scores[indices[i]], select_features[indices[i]])))


# In[24]:


dataset.head()


# In[25]:


# Dropping variables
dataset = dataset.drop(['PassengerId', 'SibSp', 'Parch','Ticket', 'Cabin'], axis = 1)


# In[26]:


dataset.head()


# In[27]:


y = dataset.Survived


# In[28]:


# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)


# In[29]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=dataset , y=y , cv = 10)
print(("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std()))


# In[30]:


# Testing data:
test = pd.read_csv('../input/test.csv')


# In[31]:


X_train = dataset
y_train = y


# In[32]:


dataset = test
# Categorizing Fare feature:

def FareFunc(data):
    data.loc[data['Fare'].isnull(), 'Fare'] = 7            
    data['FareCat'] = 0
    data.loc[data['Fare'] < 8, 'FareCat'] = 0
    data.loc[(data['Fare'] >= 8 ) & (data['Fare'] < 16),'FareCat' ] = 1
    data.loc[(data['Fare'] >= 16) & (data['Fare'] < 30),'FareCat' ] = 2
    data.loc[(data['Fare'] >= 30) & (data['Fare'] < 45),'FareCat' ] = 3
    data.loc[(data['Fare'] >= 45) & (data['Fare'] < 80),'FareCat' ] = 4
    data.loc[(data['Fare'] >= 80) & (data['Fare'] < 160),'FareCat' ] = 5
    data.loc[(data['Fare'] >= 160) & (data['Fare'] < 270),'FareCat' ] = 6
    data.loc[(data['Fare'] >= 270), 'FareCat'] = 7
FareFunc(dataset)

# Creating FamlSize Feature:
def FamlSize(data):
    data['FamlSize'] = 0
    data['FamlSize'] = data['SibSp'] + data['Parch'] + 1
FamlSize(dataset)

#Filling NA:
row_index = dataset.Embarked.isnull()
dataset.loc[row_index,'Embarked']='S' 

#Categorizing Port as int:
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
dataset.Embarked=labelEncoder_X.fit_transform(dataset.Embarked)

# Filling NA Age:
print(('Number of null values in Age:', sum(dataset.Age.isnull())))

got = dataset.Name.str.split(',').str[1]
dataset.iloc[:,2]=pd.DataFrame(got).Name.str.split('\s+').str[1]
ax = plt.subplot()
ax.set_ylabel('Average age')
dataset.groupby('Name').mean()['Age'].plot(kind = 'bar', figsize=(13,8), ax = ax)

# Filling Age by title
title_mean_age=[]
title_mean_age.append(list(set(dataset.Name))) 
title_mean_age.append(dataset.groupby('Name').Age.mean())
title_mean_age

n_traning= dataset.shape[0]   
n_titles= len(title_mean_age[1])
for i in range(0, n_traning):
    if np.isnan(dataset.Age[i])==True:
        for j in range(0, n_titles):
            if dataset.Name[i] == title_mean_age[0][j]:
                dataset.Age[i] = title_mean_age[1][j]
                
dataset["Title"] = dataset['Name']
dataset=dataset.drop(['Name'], axis=1)

#Categorizing Title as int:
dataset.Title=labelEncoder_X.fit_transform(dataset.Title)

# Categorizing Age:

def AgeCat(data):
    data['AgeCat'] = 0
    data.loc[(data['Age'] <= 5), 'AgeCat'] = 0
    data.loc[(data['Age'] <= 12) & (data['Age'] > 5), 'AgeCat'] = 1
    data.loc[(data['Age'] <= 18) & (data['Age'] > 12), 'AgeCat'] = 2
    data.loc[(data['Age'] <= 22) & (data['Age'] > 18), 'AgeCat'] = 3
    data.loc[(data['Age'] <= 32) & (data['Age'] > 22), 'AgeCat'] = 4
    data.loc[(data['Age'] <= 45) & (data['Age'] > 32), 'AgeCat'] = 5
    data.loc[(data['Age'] <= 60) & (data['Age'] > 45), 'AgeCat'] = 6
    data.loc[(data['Age'] <= 70) & (data['Age'] > 60), 'AgeCat'] = 7
    data.loc[(data['Age'] > 70), 'AgeCat'] = 8
AgeCat(dataset)

# Categorizing Sex as int:
dataset.Sex=labelEncoder_X.fit_transform(dataset.Sex)

dataset.head()


# In[33]:


dataset.head()


# In[34]:


dataset = dataset.drop(['PassengerId', 'SibSp', 'Parch','Ticket', 'Cabin','Fare'], axis = 1)


# In[35]:


dataset.head()


# In[36]:


X_train.head()


# In[37]:


X_train = X_train.drop(['Survived', 'Fare'], axis = 1)


# In[38]:


# Checking model performance:
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train , y=y_train , cv = 10)
print(("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std()))


# In[39]:


classifier.fit(X_train,y_train)


# In[40]:


predicao = classifier.predict(dataset)
predicao


# In[41]:


PassengerId =np.array(test["PassengerId"]).astype(int)
sampl = pd.read_csv('../input/gender_submission.csv')
sampl['Survived'] = pd.DataFrame(predicao)
sampl.to_csv('submission9.csv', index=False)


# In[42]:




