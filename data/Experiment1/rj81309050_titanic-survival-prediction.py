#!/usr/bin/env python
# coding: utf-8

# # Titanic : Machine learning from Disaster
# Author - Rishabh Jain

# In[1]:


import warnings,os,math
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


# ## Loading Train and Test set

# In[2]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
train.head()


# In[3]:


recordId='PassengerId'
target='Survived'
trainId=train[recordId]
testId=test[recordId]

# Dropping **PassengerId** (unique identifier) feature from train & test set.
train.drop(recordId,axis=1,inplace=True)
test.drop(recordId,axis=1,inplace=True)

# Checking Dataset shape
print('Train Set\t %d X %d'%(train.shape[0],train.shape[1]))
print('Test Set\t %d X %d'%(test.shape[0],test.shape[1]))


# ## Data Preprocessing

# In[4]:


features=['Pclass','SibSp','Parch','Sex','Embarked','Age','Fare','Survived']
nrows=2
ncols=int(np.ceil(len(features)/nrows))
fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(14,5))
fig.subplots_adjust(wspace=0.4,hspace=0.4)
for row in range(nrows):
    for col in range(ncols):
        feature=features[row*ncols+col]
        if feature in ['Age','Fare']:
            sns.violinplot(train[target],train[feature],ax=ax[row,col])
        else:
            sns.barplot(train[feature],train[target],ax=ax[row,col])            


# **Few observations from the plots :**
# - **Pclass -** An ordinal feature where passsenger with `Pclass=1` had higher probablitly of surviving than compared to passenger with `Pclass=3`.
# - **SibSp -** Passengers with lesser number of siblings and spouses had higher chances of surviving.
# - **Sex -** Females were more likely to survive in titanic disaster.
# - **Embarked -** Passengers who embarked at port C had higher probability than other passengers.
# - **Fare -** Passengers who paid higher fare or in other words passengers with higher socio-economic status had better better chance of survival.
# 
# **Let's first concatenate the train and test set for handling missing data and feature engineering**

# In[5]:


nTrain=train.shape[0]
nTest=test.shape[0]
trainY=train[target]
allData=pd.concat((train,test)).reset_index(drop=True)
allData.drop(target,axis=1,inplace=True)
print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))


# ### Handling Missing Data

# In[6]:


count=allData.isnull().sum().sort_values(ascending=False)
percentage=(allData.isnull().sum()/allData.isnull().count()).sort_values(ascending=False)*100
dtypes=allData[count.index].dtypes
missingData=pd.DataFrame({'Count':count,'Percentage':percentage,'Type':dtypes})
missingData.drop(missingData[missingData['Count']==0].index,inplace=True)
missingData.head(10)


# Since, **Cabin** feature is directly propotional to Socio-economic status of the passenger and contains the deck information. We will keep this feature by replacing the passengers with `Cabin=NaN` value replaced by `M`.

# In[7]:


idx=allData[allData['Cabin'].isnull()].index
allData.loc[idx,'Cabin']='M'


# According to the plots above, **Survival** of the passenger is not affected much by the age. We are going to drop this feature.

# In[8]:


allData.drop(columns=['Age'],inplace=True)


# After, looking at the names of the passenger with their embarkement port missing on internet, we can conclude that :
# - Icard, Miss. Amelie embarked for Southampton ([source](https://www.encyclopedia-titanica.org/titanic-survivor/)).
# - Martha Evelyn emabarked for Southampton ([source](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html)).

# In[9]:


allData[allData['Embarked'].isnull()]


# In[10]:


idx=allData[allData['Embarked'].isnull()].index
allData.loc[idx,'Embarked']='S'


# Only one passenger is there with fare missing. We will deal with this by following these steps :
# - First identify if this passenger is from train or test set using passenger name.
# - Replace the fare value with the mean fare of passengers with `Pclass=3`,`Sex=male` and `Embarked=S` in train/test set.

# In[11]:


allData[allData['Fare'].isnull()]


# In[12]:


name=allData[allData['Fare'].isnull()].Name.values[0]
dataset=train if name in train['Name'].tolist() else test
groups=dataset.groupby(['Pclass','Sex','Embarked'])['Fare'].mean().to_frame('Mean Fare')
groups


# In[13]:


idx=allData[allData['Fare'].isnull()].index
allData.loc[idx,'Fare']=groups.loc[3,'male','S'].values[0]


# Verifying if the all the missing values are dealt with.

# In[14]:


count=allData.isnull().sum().sort_values(ascending=False).to_frame(name='count')
count


# ### Feature Engineering
# Here, we are going to create few new features :
# 
# - **FamilySize -** Created by adding **SibSp** and **Parch** variables with 1.
# - **IsAlone -** Binary feature will be created by setting its value to 1 if **FamilySize** is 1 otherwise 0 value will be set.
# - **Title -** Extracted by the **Name** feature. Moreover, all the Titles whose occurrences is less than 10 are replaced with **'Misc'**.
# - **Deck -** Extracted by the **Cabin** feature.
# 
# After the new features are derived, **Name**,**Cabin** and **Ticket** features will be dropped from the dataframe.

# In[15]:


# FamilySize
allData['FamilySize']=allData['SibSp']+allData['Parch']+1
# IsAlone
allData['IsAlone']=None
idx=allData[allData['FamilySize']==1].index
allData.loc[idx,'IsAlone']=1
idx=allData[allData['FamilySize']>1].index
allData.loc[idx,'IsAlone']=0
allData['IsAlone']=allData['IsAlone'].astype(int)
# Title
allData['Title']=allData['Name'].str.extract(" ([A-Za-z]+)\.")
titleNames=(allData['Title'].value_counts()<10)
allData['Title']=allData['Title'].apply(lambda title: 'Misc' if titleNames.loc[title]==True else title)
# Deck
allData['Deck']=allData['Cabin'].str[0]
# Dropping Name, Cabin and Ticket feature
allData.drop(columns=['Name','Cabin','Ticket'],inplace=True)


# In[16]:


allData.head()


# **Let's take a look at how our new features effect the survival of a passenger in training set.**

# In[17]:


_train=allData[:nTrain]
features=['FamilySize','IsAlone','Title','Deck']
nrows=1
ncols=int(np.ceil(len(features)/nrows))
fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(14,2.5))
fig.subplots_adjust(wspace=0.4,hspace=0.4)
for col in range(ncols):
    feature=features[col]
    if feature is not 'Deck':
        sns.barplot(_train[feature],trainY,ax=ax[col])
    else:
        sns.barplot(_train[feature],trainY,ax=ax[col],order=['A','B','C','D','E','F','G','M','T'])


# **It's important to understand if the Cabin/Deck are assigned to passengers based on socio-economic status.**<br><br>
# <img src="images/Titanic_side_plan.png" style="width:700px;">

# In[18]:


_train.groupby(['Deck','Pclass']).size().to_frame(name='Passenger Count')


# From the table above, we can clearly conclude few things :
# - Deck A,B,C and T were only reserved for passenger with `Pclass=1` and will be replaced by 'ABC'.
# - Deck D & E were reserved for passengers with `Pclass=1,2,3` and will be replaced by 'DE'.
# - Deck F & G were reserved for passengers with `Pclass=2,3` and will be replace by 'FG'.
# - Deck M is just a placeholder values for passengers with no cabins.
# 
# This way, we will have reduced the cardinality of deck feature from 9 to 4.

# In[19]:


allData['Deck']=allData['Deck'].replace(['A','B','C','T'],'ABC')
allData['Deck']=allData['Deck'].replace(['D','E'],'DE')
allData['Deck']=allData['Deck'].replace(['F','G'],'FG')
allData['Deck'].value_counts()


# In[20]:


fig,ax=plt.subplots(figsize=(5,4))
corrMat=allData.corr()
sns.heatmap(corrMat,annot=True)


# **Converting Categorical variables into Dummy variables**

# In[21]:


allData=pd.get_dummies(allData)
print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))
allData.sample(5)


# **Splitting dataset back to training and test set**

# In[22]:


trainX=allData[:nTrain]
testX=allData[nTrain:]


# ## Modeling
# Train Validation Split on Training Data (for Cross Validation)

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Splitting training set further into training and validation set
subTrainX,valX,subTrainY,valY=train_test_split(trainX,trainY,test_size=0.2,random_state=42)


# Here, we will try several models:
# - Nearest Neighbors
# - Linear SVM
# - RBF SVM
# - Decision Tree
# - Random Forest 
# - Neural Net
# - Logistic Regression

# In[24]:


classifiers={
    "Nearest Neighbors":KNeighborsClassifier(3), 
    "Linear SVM":SVC(kernel='linear'), 
    "RBF SVM":SVC(kernel='rbf'),
    "Decision Tree":DecisionTreeClassifier(max_features=10,max_depth=10), 
    "Random Forest":RandomForestClassifier(n_estimators=200,max_features=10,max_depth=10), 
    "Neural Net":MLPClassifier(alpha=0.001),
    "Logistic Regression":LogisticRegression()
}


# In[25]:


# Training
models_accuracy=[]
i=0
for classifier_name,classifier in classifiers.items():
    i+=1
    print("Training classifiers{}".format("."*i),end='\r')
    classifier.fit(subTrainX,subTrainY)
    predictions=classifier.predict(subTrainX)
    train_accuracy=accuracy_score(subTrainY,predictions)
    predictions=classifier.predict(valX)
    test_accuracy=accuracy_score(valY,predictions)
    models_accuracy.append({
        'classifier':classifier_name,
        'train':train_accuracy,
        'test':test_accuracy
    })
models_accuracy=pd.DataFrame(models_accuracy)
models_accuracy=models_accuracy.sort_values(by=['test','train'],ascending=False)
df=models_accuracy.melt(
    id_vars='classifier',
    value_name='accuracy'
)
sns.barplot(x='accuracy',y='classifier',hue='variable',data=df);
models_accuracy


# Let's take a look at the TOP 10 important features for Random Forest classifier.

# In[26]:


feature_importance=pd.DataFrame({'feature':subTrainX.columns,'importance':classifiers['Random Forest'].feature_importances_})
feature_importance=feature_importance.sort_values(by='importance',ascending=False)
feature_importance.head(10)


# ### For Final Submission
# **We can conclude that amongst all the classifiers, RANDOM FOREST has the highest test accuracy. Lets train the Random Forest classifier with the ENTIRE TRAINING DATA for better predictions on final test set.**
# #### Training

# In[27]:


rfc=RandomForestClassifier(n_estimators=200,max_features=10,max_depth=10)
rfc.fit(trainX,trainY)
predictions=rfc.predict(trainX)
accuracy=accuracy_score(trainY,predictions)
print('TRAINING ACCURACY : {:.4f}'.format(accuracy))


# #### Final Prediction

# In[28]:


predictions=rfc.predict(testX)
submission=pd.DataFrame()
submission[recordId]=testId
submission[target]=predictions
submission.head()


# In[29]:


submission.to_csv('submission.csv',index=False)

