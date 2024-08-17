#!/usr/bin/env python
# coding: utf-8

# # 1.Problem Statement
# 
# * Titanic has became one of the most famous ships in the history.
# * One of the unexcepted event is the wreck of Titanic On April 15,1912 after striking an iceberg during her maiden voyage from Southampton to New York City.
# * Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# * While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# * The challenge is to find the survival status of passengers using passenger data (ie name, age, gender, socio-economic class, etc).  
# 
# 

# # 2.Overview of Dataset
# 
# There are two sets of data available in the data directory
# 
#  1.training set (train.csv) use to build machine learning model.
#  
#  2.test set (test.csv) use to prdict the built model.
#  
# 

# In[2]:


# To check data directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))



# #  3.Data Preparation

# In[17]:


#import basic libraries

import numpy as np                 # For Linear Algebra
import pandas as pd                # For data manipulation

import matplotlib.pyplot as plt    # For data visualization
import seaborn as sns              # For data visualization

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings                   #ignore warnings
warnings.filterwarnings('ignore')


# In[4]:


# Read train and test dataset

train = pd.read_csv("../input/titanic/train.csv")
test  = pd.read_csv("../input/titanic/test.csv")


# In[5]:


#make copy of train and test datset

train_df = train.copy()
test_df  = test.copy()

#Combine both datasets for running certain operations together

combine = [train_df, test_df]


# In[6]:


#View sample of top 5 records in train dataset

train_df.head()


# In[7]:


#Display sample of test dataset

test_df.head()


# In[8]:


#display the shape of train and test datset

train_df.shape,test_df.shape


# Observations:
# 
# 1.Train dataset contains 891 rows and 12 columns.
# 
# 2.Test dataset contains 418 rows and 11 columns.
# 
# 3.The objective is to train the independent variable and build statistical model to predict the the target variable.
# 
# 4.In train dataset the target variable is considered ("Survived") which contains categorical variable 0  and 1 .

# # 4.Basic Statistics

# In[9]:


# Satistical summary of numerical variable in the train datset.

train_df.describe()


# In[10]:


#Statistical summary of categorical variables in the train dataset.

train_df.describe(include=['object'])


# **Data types:**
# 
# 1. Categorical Features
# 
#        *  Categorical  : Name,Survived,Sex and Embarked 
#        *  Ordinal      : Pclass
#        
# 1. Numerical Features 
# 
#        *  Continous Variable : Age and Fare
#        *  Discrete Variable  : SibSp and Parch 
#        
# 1. Alpha Numeric  : Ticket and Cabin
# 
#        

# In[11]:


train_df.info()


# # 5.Univariate Analysis

# In[18]:


train_df['Survived'].value_counts().plot.bar()


# * The bar chart explains less number of passengers survived in the shipwrecks.

# In[16]:


train_df['Pclass'].value_counts().plot.bar()


# * The variable Pcclass illustrates largest number of passengers booked 3rd class ticket.
# 
# * The second largest number of passengers booked 1st class ticket and remaining passengers booked by 2nd class ticket.
# 
# * In addition to that if more number of passengers travelled in 3 rd class are lesser chance for survival.

# In[17]:


train_df['SibSp'].value_counts().plot.bar()


# In[18]:


train_df['Parch'].value_counts().plot.bar()


# * The variable Sibsp indicates number of passengers on boarded with their siblings and spouses.
# * The variable Parch recorded number of passengers on boarded with their parents and children.
# * Both variable are following the similar shape of distributions.
# * Larger number of people on boarded as single passengers. 

# In[19]:


train_df['Sex'].value_counts().plot.bar()


# * The shortest bar indicates that less number of female passengers onboarded in RMS Titanic.

# In[13]:


train_df['Embarked'].value_counts(normalize=True).plot.bar()


# * Around 70% of passengers embarked from S(Southampton) port.
# * Around 20% of passengers embarked from C(Cherbourg) port and remaining passengers embarked from C (Cherbourg) port.

# In[14]:


sns.distplot(train_df['Fare'],color="m", ) 


# * The distribution of Fare variable is right skewed which means more number of passengers chosen 3 rd class where the ticket price is comparetevely less as compared to 2nd and 1 st class.

# In[15]:


plt.subplot(121) 
sns.distplot(train_df['Age'],color="m", ) 

plt.subplot(122) 
train_df['Age'].plot.box(figsize=(16,5)) 

plt.show()


# * The Age variable follows nomal distribution and the maximum age of passenger was 80 years old.
# 

# # 6.Bivariate Analysis

# In[16]:


#draw a bar plot of survival by sex

sns.barplot(x='Sex',y="Survived",data=train_df)

#print percentages of females vs. males that survive

print(("Percentage of females who survived:", train_df["Survived"][train_df["Sex"] == 'female'].value_counts(normalize = True)[1]*100))

print(("Percentage of males who survived:", train_df["Survived"][train_df["Sex"] == 'male'].value_counts(normalize = True)[1]*100))


# * It is intersting to see that around 74% of female passengers survived.
# * This information conveys a larger number of male passengers were not evacuated because the protocol follws "children and women" loaded first on the lifeboats.

# In[19]:


#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train_df)

#print percentage of people by Pclass that survived
print(("Percentage of Pclass = 1 who survived:", train_df["Survived"][train_df["Pclass"] == 1].value_counts(normalize = True)[1]*100))

print(("Percentage of Pclass = 2 who survived:", train_df["Survived"][train_df["Pclass"] == 2].value_counts(normalize = True)[1]*100))

print(("Percentage of Pclass = 3 who survived:", train_df["Survived"][train_df["Pclass"] == 3].value_counts(normalize = True)[1]*100))


# * Passengers travelled in the first class accommodation has highest percentage of survival followed by 2nd and 3rd class passengers. 

# In[20]:


#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train_df)

# printing individual percent values for all of these.
print(("Percentage of SibSp = 0 who survived:", train_df["Survived"][train_df["SibSp"] == 0].value_counts(normalize = True)[1]*100))

print(("Percentage of SibSp = 1 who survived:", train_df["Survived"][train_df["SibSp"] == 1].value_counts(normalize = True)[1]*100))

print(("Percentage of SibSp = 2 who survived:", train_df["Survived"][train_df["SibSp"] == 2].value_counts(normalize = True)[1]*100))


# * The order of survival is higher for passengers with one,two and no siblings/spouses.
# * Remaining passengers have less survival rate.

# In[21]:


#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train_df)
plt.show()


# * Passengers with parents/children has highest survival rate in number 3 as compared to number 1 and 2. 

# In[22]:


#sort the ages into logical categories

train_df["Age"] = train_df["Age"].fillna(-0.5)
test_df["Age"] = test_df["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train_df['AgeGroup'] = pd.cut(train_df["Age"], bins, labels = labels)
test_df['AgeGroup'] = pd.cut(test_df["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train_df)
plt.show()


# * The Age group seems like Baby category has higher survival rate.
# * The survival rate is same for the category Young Adult,Adult. 
# * The second highest survival rate is teenager as compared to child and student.

# In[23]:


train_df.head()


# # 7.Missing Values Treatment

# In[24]:


pd.isnull(train_df).sum()


# In[30]:


pd.isnull(test_df).sum()


# In[33]:


train_1 = train_df.drop(['Cabin'], axis = 1)
test_1 = test_df.drop(['Cabin'], axis = 1)


# In[34]:


train_1.shape,test_1.shape


# In[35]:


train_2 = train_1.drop(['Ticket'], axis = 1)
test_2 = test_1.drop(['Ticket'], axis = 1)


# In[36]:


train_2.shape,test_2.shape


# In[37]:


#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")
southampton = train_2[train_2["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train_2[train_2["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train_2[train_2["Embarked"] == "Q"].shape[0]
print(queenstown)


# In[38]:


#replacing the missing values in the Embarked feature with S
train_2 = train_2.fillna({"Embarked": "S"})


# In[39]:


#create a combined group of both datasets
combine = [train_2, test_2]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_2['Title'], train_2['Sex'])


# In[40]:


#replace various titles with more common names

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_2[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[41]:


#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_2.head()


# In[42]:


# fill missing age with mode age group for each title
mr_age = train_2[train_2["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train_2[train_2["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train_2[train_2["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train_2[train_2["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train_2[train_2["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train_2[train_2["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#I tried to get this code to work with using .map(), but couldn't.
#I've put down a less elegant, temporary solution for now.
#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(train_2["AgeGroup"])):
    if train_2["AgeGroup"][x] == "Unknown":
        train_2["AgeGroup"][x] = age_title_mapping[train_2["Title"][x]]
        
for x in range(len(test_2["AgeGroup"])):
    if test_2["AgeGroup"][x] == "Unknown":
        test_2["AgeGroup"][x] = age_title_mapping[test_2["Title"][x]]


# In[44]:


#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train_2['AgeGroup'] = train_2['AgeGroup'].map(age_mapping)
test_2['AgeGroup'] = test_2['AgeGroup'].map(age_mapping)

train_2.head()

#dropping the Age feature for now, might change
train_3 = train_2.drop(['Age'], axis = 1)
test_3 = test_2.drop(['Age'], axis = 1)


# In[45]:


train_3.head()


# In[46]:


test_3.head()


# In[47]:


#drop the name feature since it contains no more useful information.
train_4 = train_3.drop(['Name'], axis = 1)
test_4 = test_3.drop(['Name'], axis = 1)


# In[48]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train_4['Sex'] = train_4['Sex'].map(sex_mapping)
test_4['Sex'] = test_4['Sex'].map(sex_mapping)

train_4.head()


# In[49]:


#map each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_4['Embarked'] = train_4['Embarked'].map(embarked_mapping)
test_4['Embarked'] = test_4['Embarked'].map(embarked_mapping)

train_4.head()


# In[50]:


#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test_4["Fare"])):
    if pd.isnull(test_4["Fare"][x]):
        pclass = test_4["Pclass"][x] #Pclass = 3
        test_4["Fare"][x] = round(train_4[train_4["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train_4['FareBand'] = pd.qcut(train_4['Fare'], 4, labels = [1, 2, 3, 4])
test_4['FareBand'] = pd.qcut(test_4['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train_5 = train_4.drop(['Fare'], axis = 1)
test_5 = test_4.drop(['Fare'], axis = 1)


# In[51]:


train_5.head()


# In[52]:


test_5.head()


# # 8.Model Building

# In[62]:


from sklearn.model_selection import train_test_split

predictors = train_5.drop(['Survived', 'PassengerId','AgeGroup'], axis=1)
target = train_5["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# In[63]:


predictors.shape,target.shape


# In[64]:


x_train.shape, x_val.shape, y_train.shape, y_val.shape


# In[65]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[66]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[67]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[68]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[69]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[70]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[71]:


# Random Forest

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[72]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[73]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[74]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[75]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# In[78]:


#set ids as PassengerId and predict survival 
ids = test_5['PassengerId']
predictions = gbk.predict(test_5.drop(['PassengerId','AgeGroup'], axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# # 9.Cross Validation

# In[81]:


from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import cross_val_score


skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= 1)

Log_skf = LogisticRegression()

Log_skf1 = cross_val_score(Log_skf, x_train, y_train, cv=skfold)

print(Log_skf1)

acc_log2 = Log_skf1.mean()*100.0

acc_log2 


# In[82]:


skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= 1)

svc_sk = SVC(gamma='auto')

svc_sk1 = cross_val_score(svc_sk, x_train, y_train, cv=skfold)

print(svc_sk1)

acc_svc2 = svc_sk1.mean()*100.0

acc_svc2 


# In[83]:


skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= 1)

knn_sk = KNeighborsClassifier(n_neighbors = 3)

knn_sk1 = cross_val_score(knn_sk, x_train, y_train, cv=skfold)

print(knn_sk1)

acc_knn2 = knn_sk1.mean()*100.0

acc_knn2


# In[84]:


skfold = StratifiedKFold (n_splits=10,shuffle=False, random_state= 1)

rfc_sk = RandomForestClassifier(n_estimators=100,random_state = 1)

rfc_sk1 = cross_val_score(rfc_sk, x_train, y_train, cv=skfold)

print(rfc_sk1)

acc_rfc2 = rfc_sk1.mean()*100.0

acc_rfc2


# In[85]:


skfold = StratifiedKFold (n_splits=5,shuffle=False, random_state= 1)

gnb_sk = GaussianNB()

gnb_sk1 = cross_val_score(gnb_sk,x_train, y_train, cv=skfold)

print(gnb_sk1)

acc_gnb2 = gnb_sk1.mean()*100.0

acc_gnb2


# In[86]:


skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= 1)

ptn_sk = Perceptron()

ptn_sk1 = cross_val_score(ptn_sk, x_train, y_train, cv=skfold)

print(ptn_sk1)

acc_ptn2 = ptn_sk1.mean()*100.0

acc_ptn2


# In[87]:


skfold = StratifiedKFold (n_splits=5,shuffle=True, random_state= None)

dt_sk = DecisionTreeClassifier(random_state=1)

dt_sk1 = cross_val_score(dt_sk, x_train, y_train, cv=skfold)

print(dt_sk1)

acc_dt2 = dt_sk1.mean()*100.0

acc_dt2


# In[88]:


import lightgbm as lgb

skfold = StratifiedKFold (n_splits=10,shuffle=True, random_state= None)

lgb_sk = lgb.LGBMClassifier()

lgb_sk1 = cross_val_score(lgb_sk, x_train, y_train, cv=skfold)

print(lgb_sk1)

acc_lgb2 = lgb_sk1.mean()*100.0

acc_lgb2


# # 10.Model Evaluation

# In[91]:


models = pd.DataFrame({
    'Model': ['Support Vector Classifier', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron',  
              'Decision Tree','LGBMClassifier'],
    'Mean Accuracy': [acc_svc2, acc_knn2, acc_log2, 
              acc_rfc2, acc_gnb2, acc_ptn2, 
             acc_dt2,acc_lgb2]})
models.sort_values(by='Mean Accuracy', ascending=False)


# * After using cross validation our model SVC predicts 83% mean accuracy.This is my first attempt and I believe that still there are ways to improve the model prediction using feature engineering.
# 
# * Any suggestions to improve our score are most welcome.

# Reference:
# 
# This notebook has been created based on great work done solving the Titanic competition and other sources.
# 
# 1. Titanic Data Science Solutions
