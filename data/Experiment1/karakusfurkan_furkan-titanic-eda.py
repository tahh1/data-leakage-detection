#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# The sinking of Titanic is one of the most notorious shipwrecks in the history. In 1912, during her voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew.
# <br>
# <font color = 'blue'>
# Content:
# <br>
# 1. [Load and Check Data](#1)
# <br>
# 2. [Variable Description](#2)<br>
#     &ensp;2.1 [Univariate Variable Analysis](#3)<br>
#     &ensp;2.2 [Categorical Variable Analysis](#4)<br>
#     &ensp;2.3 [Numerical Variable Analysis](#5)<br>
# <br>
# 3. [Basic Data Analysis](#6)
# <br>
# 4. [Outlier Detection](#7)<br>
# 5. [Missing Value](#8) <br>
#   &ensp;  5.1 [Find Missing Value](#9)<br>
#  &ensp;   5.2 [Fill Missing Value](#10)<br>
# 6. [Visualization](#11)<br>
#     &ensp; 6.1 [Correlation Between Sibsp -- Parch -- Age -- Fare -- Survived](#12)<br>
#     &ensp; 6.2 [SibSp--Survived](#13)<br>
#     &ensp; 6.3 [Parch--Survived](#14)<br>
#     &ensp; 6.4 [Pclass--Survived](#15)<br>
#     &ensp; 6.5 [Age--Survived](#16)<br>
#     &ensp; 6.6 [Pclass--Age--Survived](#17)<br>
#     &ensp; 6.7 [Embarked--Pclass--Sex--Survived](#18)<br>
#     &ensp; 6.8 [Embarked--Fare--Sex--Survived](#19)<br>
#     &ensp; 6.9 [Fill Missing: Age Feature](#20)<br>
# 7. [Feature Engineering](#21)<br>
#     &ensp; 7.1 [Name--Title](#22)<br>
#     &ensp; 7.2 [Family Size](#23)<br>
#     &ensp; 7.3 [Embarked](#24)<br>
#     &ensp; 7.4 [Ticket](#25)<br>
#     &ensp; 7.5 [Pclass](#26)<br>
#     &ensp; 7.6 [Sex](#27)<br>
#     &ensp; 7.7 [Dropping Passenger ID and Cabin](#28)<br>
# 8. [Modeling](#29)<br>
#     &ensp; 8.1 [Train-Test Split](#30)<br>
#     &ensp; 8.2 [Simple Logistic Regression](#31)<br>
#     &ensp; 8.3 [Hyperparameter Tuning -- Grid Search -- Cross Validation](#32)<br>
#     &ensp; 8.4 [Ensemble Modeling](#33)<br>
#     &ensp; 8.5 [Prediction and Submission](#34)<br>
# 
# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# <a id ="1">
# ## Load and Check Data 

# In[2]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"]


# In[3]:


train_df.head()


# In[4]:


train_df.describe()


# <a id = "2" >
# # Variable Description
# 
# 1. PassengerId	:  unique id number to each passenger
# 1. Survived	 :  passenger survived(1) or died(0)
# 1. Pclass	:  passenger class
# 1. Name	 :  passenger name
# 1. Sex	:  gender of passenger
# 1. Age	:  age of passenger
# 1. SibSp	:  number of siblings/spouses
# 1. Parch	:  number of parents/children
# 1. Ticket	:  ticket number
# 1. Fare	 :  amount of money spent on ticket
# 1. Cabin	:  cabin category
# 1. Embarked  :  port where passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[5]:


train_df.info()


# * float64(2) : Fare ve Age
# * int64(5) : Pclass, sibsp, parch, passengerId and survived
# * object(5) : Cabin, embarked, name, sex and ticket

# <a id = '3'>
# # Univariate Variable Analysis
# ### Categorical Variable :
# *Survived,
# *Sex,
# *Pclass,
# *Embarked,
# *Cabin,
# *Name,
# *Ticket,
# *Sibsp,
# Parch
# ### Numerical Variable :
# *Fare,
# *Age,
# PassengerId
# 

# <a id = '4'>
# ### Categorical Variable Analysis
# 

# In[6]:


def bar_plot(variable):
    """
        input : variable ex: "Sex"
        output : bar plot & value count
    """
    #getting feature
    var = train_df[variable]
    
    #count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    #visualize
    
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print(("{}: \n {}".format(variable,varValue)))
    


# In[7]:


category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp","Parch"]
for c in category1:
    bar_plot(c)


# <a id = '5'>
# ### Numerical Variable Analysis

# In[8]:


def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distrubiton with histogram".format(variable))
    plt.show()
    


# In[9]:


numericVar = ["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)


# <a id = '6'> 
# ## Basic Data Analysis
# 
# 1. Pclass - Survived
# 1. Sex - Survived
# 1. SibSp - Survived
# 1. Parch - Survived

# In[10]:


#Pclass vs Survived
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# In[11]:


#Sex vs Survived
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# In[12]:


#SibSp vs Survived
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# In[13]:


train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived",ascending = False)


# <a id = "7">
# # Outlier Detection

# In[14]:


def det_outlier(df,features):
    outlier_indices = []
    
    for c in features:
        #1st quartile
        Q1 = np.percentile(df[c],25)
        
        #3rd quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3-Q1 
        #Outlier step
        outlier_step = IQR * 1.5
        
        #detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q1 + outlier_step)].index
        
        # store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    
    multiple_outliers = list(i for i, v in list(outlier_indices.items()) if v >2) #study it
    
    return multiple_outliers


# In[15]:


train_df.loc[det_outlier(train_df, ["Age","SibSp","Parch","Fare"])]


# In[16]:


# drop outliers
train_df = train_df.drop(det_outlier(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)


# In[17]:


train_df


# <a id = "8">
# ### Missing Value
#   &ensp; Find Missing Value<br>
#   &ensp; Fill Missing Value

# In[18]:


#fitting train and test data frames
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis= 0).reset_index(drop = True)


# In[19]:


train_df.head()


# <a id = "9">
# ## Find Missing Value

# In[20]:


#finding missing values
train_df.columns[train_df.isnull().any()]



# In[21]:


#finding count missing values
train_df.isnull().sum()


# <a id = "10">
# ## Fill Missing Value
# * Embark has 2 missing values
# * Fare has 1 missing value

# In[22]:


train_df[train_df["Embarked"].isnull()]


# In[23]:


#firstly embarked
train_df.boxplot(column = "Fare", by = "Embarked")
plt.show()


# In[24]:


#I looked the ports and saw C port has most likely right for these 2 person.
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df[train_df["Embarked"].isnull()]


# In[25]:


#now fare
np.mean(train_df[train_df["Pclass"] == 3]["Fare"])


# In[26]:


#filling with the mean value.
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))


# In[27]:


train_df[train_df["Fare"].isnull()]


# <a id = "11">
# # Visualization
# 

# <a id="12">
# Correlation Between Sibsp -- Parch -- Age -- Fare -- Survived
# </a>

# In[28]:


#seaborn library used
#corr method is correlation
#annot = True --> you can see numbers.
list1 = ["SibSp","Parch","Age","Fare","Survived"]
sns.heatmap(train_df[list1].corr(),annot = True, fmt = ".2f")
plt.show()


# <font color = "purple">Fare feature seems to have correlation with survived feature (0.26) which means if passenger had paid a lot for ticket, he/she could survive more likely.
#     </font>

# <a id = "13">
# SibSp -- Survived

# In[29]:


g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df ,kind = "bar", size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# * Having a lot of SibSp have less chance to survive.
# * If sibsp == 0 or 1 or 2, passengers have more chance to survive
# * we can consider a new feature describing these categories.
# 

# <a id = "14">
# Parch -- Survived

# In[30]:


g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Survived Probability")
plt.show()


# * Note that Parch(3) values are not stable. parch(3) = huge std (standard sapma)
# * Sibsp and Parch can be use for new feature extraction with th = 3
# * small families have more chance to survive.
# 

# <a id = "15">
# Pclass -- Survived
# 

# In[31]:


g = sns.factorplot(x= "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)


# that feature is so obvious

# <a id = "16">
# Age -- Survived
# 

# In[32]:


g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# * age <= 10 has a a high survival rate
# * oldest passengers (80) survived
# * large number of 20 years old didn't survive
# * most passengers are in 15-35 age range
# * we can use age feature in training
# * we can use age distrubition for missing value of age

# <a id = "17">
# Pclass--Age--Survived

# In[33]:


g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 3)
g.map(plt.hist,"Age", bins = 25)
g.add_legend()
plt.show()


# P class is important feature for training.

# <a id = "18"> 
#     Embarked--Pclass--Sex--Survived

# In[34]:


g = sns.FacetGrid(train_df, row = "Embarked", size = 3)
g.map(sns.pointplot,"Pclass","Survived","Sex")
g.add_legend()
plt.show()


# * Female passengers have much better survival rate than male ones.
# * Males have better survival rate in pclass = 3 in C port.
# * Embarked and sex will be use in training.

# <a id = "19"> 
#     Embarked--Fare--Sex--Survived

# In[35]:


g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 3)
g.map(sns.barplot,"Sex","Fare")
g.add_legend()
plt.show()


# * Passengers who paid higher fare have better survival rate.
# * Fare can be use as categorical feature for training.

# <a id = "20">
# ### Fill Missing: Age Feature

# In[36]:


train_df[train_df["Age"].isnull()]


# In[37]:


#trying with sex
sns.factorplot(x = "Sex", y = "Age", data= train_df, kind = "box")
plt.show()


# <font color = "purple"> 
# Sex is not informative for age prediction, age distribution seems to be same.

# In[38]:


#trying with Pclass
sns.factorplot(x = "Sex", y = "Age",hue = "Pclass", data= train_df, kind = "box")
plt.show()


# <font color = "red">
# Pclass is very informative about guessing for age. 1st class passengers are older than 2sn, and 2nd class passengers are older than 3rd class ones.

# In[39]:


#trying with parch and sibsp
sns.factorplot(x = "Parch", y = "Age", data= train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data= train_df, kind = "box")
plt.show()


# In[40]:


train_df["Sex"] = [1 if i == "male" else 0 for i in train_df["Sex"]]


# In[41]:


sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),annot = True)
plt.show()


# <font color ="green">
# Age is not correlated with sex but it is correlated with parch, sibsp and pclass

# In[42]:


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)


# In[43]:


for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"]) & (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_median = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_median


# In[44]:


train_df[train_df["Age"].isnull()]


# <a id ="21">
# # Feature Engineering

# <a id="22">
#     Name--Title
#     

# In[45]:


name = train_df["Name"]
#split before and after "." then take first one [0] then split before and after "," then take title "mr" or "mrs"
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]


# In[46]:


train_df["Title"].head(10)


# In[47]:


sns.countplot(x= "Title", data= train_df)
plt.xticks(rotation = 60)
plt.show()


# In[48]:


#convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i== "Mr" else 3 for i in train_df["Title"]]


# In[49]:


sns.countplot(x= "Title", data= train_df)
plt.xticks(rotation = 60)
plt.show()


# In[50]:


g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Survival Probability")
plt.show()


# In[51]:


train_df.drop(labels = ["Name"] , axis = 1 ,inplace = True)


# In[52]:


train_df.head(5)


# In[53]:


train_df = pd.get_dummies(train_df, columns=["Title"])


# In[54]:


train_df.head(10)


# <a id = "23">
# ## Family Size

# In[55]:


#We will combine to parch and sibsp to new feature named Family Size
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1


# In[56]:


train_df.head(9)


# In[57]:


#correlation between fsize with survived
g = sns.factorplot(x = "Fsize", y= "Survived", data = train_df, kind = "bar")
g.set_ylabels("Survival")
plt.show()


# In[58]:


train_df["family_size"] = [1 if i <5 else 0 for i in train_df["Fsize"]]


# In[59]:


train_df.head(10)


# In[60]:


sns.countplot(x = "family_size", data = train_df)
plt.show()


# In[61]:


g = sns.factorplot(x ="family_size", y = "Survived", data = train_df , kind =  "bar")
g.set_ylabels("Survived Probability")
plt.show()


# Smaller families have more chance to survive than larger families.

# In[62]:


train_df = pd.get_dummies(train_df, columns = ["family_size"])


# In[63]:


train_df.head(15)


# <a id = "24">
# ### Embarked

# In[64]:


sns.countplot(x = "Embarked", data = train_df)
plt.show()


# In[65]:


train_df = pd.get_dummies(train_df, columns = ["Embarked"])


# In[66]:


train_df.head(5)


# <a id = "25">
# ### Ticket Feature

# In[67]:


train_df["Ticket"].head(20)


# In[68]:


tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")

train_df["Ticket"] = tickets


# In[ ]:





# In[69]:


train_df = pd.get_dummies(train_df , columns = ["Ticket"],prefix = "T")


# In[70]:


train_df.head(10)


# <a id ="26">
# ### Pclass

# In[71]:


sns.countplot(x = "Pclass", data = train_df)
plt.show()


# In[72]:


train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df , columns = ["Pclass"])
train_df.head(10)


# <a id = "27">
# ### Sex

# In[73]:


train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df , columns = ["Sex"])
train_df.head(10)


# <a id ="28">
# ### Dropping Passenger ID and Cabin

# In[74]:


train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)


# In[75]:


train_df.columns


# <a id = "29">
# # Modeling

# In[76]:


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# <a id = "30">
# ## Train-Test Split

# In[77]:


train_df_len


# In[78]:


test = train_df[train_df_len:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)


# In[79]:


test.head()


# In[80]:


train = train_df[:train_df_len]
X_train = train.drop(labels= "Survived", axis = 1)
y_train = train["Survived"]
X_train , X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.33,random_state = 42)
print(("X_train",len(X_train)))
print(("X_test",len(X_test)))
print(("y_test",len(y_test)))
print(("y_train",len(y_train)))


# In[81]:


print(("test",len(test)))


# <a id = "31">
# ### Simple Logistic Regression

# In[82]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train,y_train)* 100,2)
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print(("Training Accuracy: {}".format(acc_log_train)))
print(("Testing Accuracy: {}".format(acc_log_test)))


# <a id = "32">
# ## Hyperparameter Tuning -- Grid Search -- Cross Validation
# <br>
# We will compare 5 machine learning classifier and evaluate mean accuracy of each of them by stratified cross validation. These are:
# * Decision Tree
# * SVM
# * Random Forest
# * KNN
# * Logistic Regression

# In[83]:


random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
              SVC(random_state = random_state),
              RandomForestClassifier(random_state = random_state),
              LogisticRegression(random_state = random_state),
              KNeighborsClassifier()
             ]


# In[84]:


dt_param_grid = {"min_samples_split": list(range(10,500,20)),
                "max_depth": list(range(1,20,2))}
svc_param_grid = {"kernel": ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}
rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf": [1,3,10],
                "bootstrap": [False],
                "n_estimators": [100,300],
                "criterion": ["gini"]}
logreg_param_grid = {"C": np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype= int).tolist(),
                 "weights":["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                    svc_param_grid,
                    rf_param_grid,
                    logreg_param_grid,
                    knn_param_grid]


# In[85]:


cv_results = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid = classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1, verbose = 1)
    clf.fit(X_train,y_train)
    cv_results.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print((cv_results[i]))
    


# In[86]:


cv_result_bar = pd.DataFrame({"Cross Validation Means": cv_results, "ML Models": ["DecisionTreeClassifier", "SVM",
                                                                                 "RandomForestClassifier",
                                                                                 "LogisticRegression",
                                                                                 "KNeighborsClassifier"]})
g= sns.barplot("Cross Validation Means", "ML Models", data = cv_result_bar)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
plt.show()


# <a id = "33">
# ## Ensemble Modeling

# In[87]:


votingC = VotingClassifier(estimators = [("dt", best_estimators[0]),
                                        ("rfc", best_estimators[2]),
                                        ("lr", best_estimators[3])],
                                        voting = "soft", n_jobs = -1)

#voting = "hard" is making simpler operation.
votingC = votingC.fit(X_train,y_train)
print((accuracy_score(votingC.predict(X_test),y_test)))


# <a id = "34">
# ## Prediction and Submission

# In[88]:


test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)
results = pd.concat([test_PassengerId,test_survived], axis = 1)
results.to_csv("titanic.csv", index = False)

