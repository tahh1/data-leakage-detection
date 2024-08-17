#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score,roc_curve
import warnings
warnings.filterwarnings('ignore')
import joblib


# In[2]:


df=pd.read_csv('titanic.csv')
df


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# so we can see that there are nan values present in the dataset

# # EDA and Visualization

# In[6]:


df


# # from the above dataset our target variable is Survived column 
# 
# # cabin column has a lot of nan values as well as it is not very helpful to predict survival resluts
# 
# # Also columns like Name, PassengerID and ticket can be dropped as it only acts as a identity of the passenger and logicaly it is not needed to predict survival results
# 

# In[7]:


df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[8]:


df


# In[9]:


df.dtypes


# In[10]:


df.isnull().sum()


# In[11]:


df['Age']=df['Age'].fillna(df['Age'].mean())


# In[12]:


df['Embarked'].value_counts()


# In[13]:


df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[14]:


df.isnull().sum()


# In[15]:


le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])
df


# we have filled all the nan values

# In[16]:


sns.countplot(df['Survived'])


# its unbalanced data we will rebalance the dataset after cleansing

# In[17]:


sns.displot(x='Survived',col='Pclass',hue='Sex',data=df,kind='ecdf')


# In[18]:


sns.catplot(x='Pclass',y='Survived',hue='Sex',data=df,kind='bar')


# In[19]:


sns.catplot(x='Pclass',y='Survived',hue='Sex',data=df,kind='violin')


# we can see that the females are given more importance for survival during the sinking of the ship
# 
# where sex(male=1 and female=0)

# In[20]:


for i in df.columns:
    plt.figure(figsize=(8,5))
    sns.distplot(df[i])


# In[38]:


sns.barplot(x='Parch',y='Survived',data=df)


# for higher Parch there is less chances of survival

# In[39]:


sns.barplot(x='SibSp',y='Survived',data=df)


# HIgher the sibsp lesser is the survival chance

# In[40]:


sns.barplot(x='Pclass',y='Survived',data=df)


# for pclass 1 there is a high chance of survival compared to pclas 3

# In[41]:


df.skew()


# we can see there is skewness present in the data

# checking outliers

# In[22]:


df.plot(kind='box',subplots=True,layout=(2,6),figsize=(10,10))


# # we can see there are some outliers present in the data
# 
# # Sibsp Parch & Age_group have outliers but categorical outliers don’t really exist without context. Hence outliers are valid in categorical data

# In[23]:


plt.figure(figsize=(15,10))
corr=df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm',linewidth=0.5)


# # Balancing the dataset

# In[24]:


df


# In[25]:


dfx=df.iloc[:,1:]


# In[26]:


dfx


# In[27]:


dfy=df['Survived']


# In[28]:


dfy


# In[29]:


from imblearn.over_sampling import SMOTE

sm=SMOTE()
x,y=sm.fit_resample(dfx,dfy)


# In[30]:


x.shape


# In[31]:


y.value_counts()


# # now the data set is balanced

# In[ ]:





# # Removing Skewness

# In[32]:


from sklearn.preprocessing import power_transform
x=power_transform(x,method=('yeo-johnson'))
x


# # feature scaling

# In[33]:


sc=StandardScaler()
x=sc.fit_transform(x)
x


# # building models

# In[34]:


lg=LogisticRegression()
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=42)
lg.fit(xtrain,ytrain)
lg.score(xtrain,ytrain)


# In[35]:


lg=LogisticRegression()
dtc=DecisionTreeClassifier()
knn=KNeighborsClassifier()
rfc=RandomForestClassifier() 
svc=SVC()
abc=AdaBoostClassifier()
gb=GradientBoostingClassifier()
rd=RidgeClassifier()
sgdc=SGDClassifier()

model=[lg,dtc,knn,rfc,svc,abc,gb,rd,sgdc]


# In[47]:


for m in model:
    m.fit(xtrain,ytrain)
    m.score(xtrain,ytrain)
    pred=m.predict(xtest)
    print(('Accuracy score of ',m,'is :'))
    print((accuracy_score(ytest,pred)))
    print((confusion_matrix(ytest,pred)))
    print((classification_report(ytest,pred)))
    print(('AUC ROC score :',roc_auc_score(ytest,pred)))
    print('\n')


# # from above we will take top 3 models for cross validation

# In[42]:


svc=SVC()
abc=AdaBoostClassifier()
gb=GradientBoostingClassifier()


# Support vector classifier

# In[43]:


#accuracy score of svc is 86%
scores=cross_val_score(svc,x,y,cv=5)
print(scores)
print((scores.mean()))


# AdaBoostCLassifier

# In[44]:


#accuracy score of adaboost is 85%
scores=cross_val_score(abc,x,y,cv=5)
print(scores)
print((scores.mean()))


# In[45]:


#accuracy sscores=cross_val_score(abc,x,y,cv=5)
scores=cross_val_score(gb,x,y,cv=5)
print(scores)
print((scores.mean()))


# # from above we can see that all models are having less differnece so will check after hyper tuning

# # hyper parameter tuning

# Gradient boosting tuning

# In[52]:


parameters={'n_estimators':[10,100,500],'criterion':['friedman_mse', 'mse', 'mae'],'loss':['deviance','exponential']}
clf=GridSearchCV(GradientBoostingClassifier(),parameters,cv=5,scoring='roc_auc')
clf.fit(x,y)
clf.best_params_


# In[65]:


gb=GradientBoostingClassifier(criterion='friedman_mse',loss= 'exponential', n_estimators= 100)
gb.fit(xtrain,ytrain)
gb.score(xtrain,ytrain)
pred=gb.predict(xtest)

print((accuracy_score(ytest,pred)))
print((confusion_matrix(ytest,pred)))
print((classification_report(ytest,pred)))
print(('AUC ROC score :',roc_auc_score(ytest,pred)))
print('\n')


# SUPPORT VECTOR CLASSIFIER

# In[58]:


parameters={'kernel':['poly','rbf','sigmoid'],'gamma':['sclae','auto'],'max_iter':list(range(3,11))}
clf=GridSearchCV(SVC(),parameters,cv=5,scoring='roc_auc')
clf.fit(x,y)
clf.best_params_


# In[63]:


svc=SVC(gamma='auto',kernel='poly',max_iter=8)
svc.fit(xtrain,ytrain)
pred=svc.predict(xtest)
print((accuracy_score(ytest,pred)))
print((confusion_matrix(ytest,pred)))
print((classification_report(ytest,pred)))
print(('AUC ROC score :',roc_auc_score(ytest,pred)))
print('\n')


# # we can conlude that gradient boosting classifier is the best model 

# In[67]:


fpr, tpr, threshold = roc_curve(ytest,pred)
auc = roc_auc_score(ytest,pred)
plt.plot(fpr, tpr, color ='orange', label ='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label ='ROC curve (area = %0.3f)'% auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# # Save model

# In[68]:


joblib.dump(gb,'titanic.obj')


# In[ ]:




