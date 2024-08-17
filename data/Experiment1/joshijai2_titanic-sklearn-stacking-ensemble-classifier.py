#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


#Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


# In[3]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import CatBoostEncoder


# ## Let's start

# In[4]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")


# In[5]:


train.head()


# In[6]:


print('Concise summary of training data:\n')
print((train.info()))
print(('-'*50,'\nConcise summary of testing data:\n'))
print((test.info()))


# In[7]:


#dropping PassengerId as it provides no purpose.
train.drop('PassengerId',1,inplace=True)
PID = test.PassengerId
test.drop('PassengerId',1,inplace=True)
#We will work through rest of the columns.


# ## Cleaning, Analysis and Feature Engineering

# ### ***Let's analyse categorical features***

# In[8]:


# categorical data
train.select_dtypes("object").columns


# *Though the dtype is int64 but Pclass and Survived are also categorical.*

# ***Survived - Our dependent variable***

# In[9]:


print("Survival Percentage in Titanic:")
print((round(train.Survived.sum()*100/len(train),2)))
sns.countplot(y=train.Survived, orient='h', palette='Accent')


# *Let's iterate through rest of the columns.*
# 
# ***Pclass***

# In[10]:


sns.countplot(train.Pclass, hue=train.Survived)


# It's obvious and verified through the visualization that the survival rate of `first class` was **high**.
# While survival rate of the `third class` passengers was **very low**.

# In[11]:


#let's extract class 3 and drop the column Pclass.
train['isPclass3'] = train['Pclass']==3
test['isPclass3'] = test['Pclass']==3


# ***Name***

# In[12]:


print((train.Name.sample(10)))
print(('\nTotal no of unique names:',train.Name.nunique()))


# As we can see every name is unique and therefore it wouldn't provide any help in our prediction models unless we engineer some stuff out of it such as **title** (Mr., Mrs, etc). Also tried extracting last name but didn't provide any help to our model.

# In[13]:


#Extracting title from name
title = train.Name.str.extract('([A-Za-z]+)\.')
ttitle = test.Name.str.extract('([A-Za-z]+)\.')
plt.xticks(rotation=90)
sns.countplot(title[0], palette='tab20', hue=train.Survived)
plt.xlabel('Title')


# In[14]:


title = title[0].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'Countess', 'Dona'], 'Miss/Mrs/Ms')
ttitle = ttitle[0].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'Countess', 'Dona'], 'Miss/Mrs/Ms')

title = title.replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Misc')
ttitle = ttitle.replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Misc')


# In[15]:


train = train.join(title).rename(columns = {0:'Title'}) #append the generated feature to our dataset
test = test.join(ttitle).rename(columns = {0:'Title'}) #append the generated feature to our dataset


# In[16]:


plt.xticks(rotation=90)
sns.countplot('Title',data=train,hue='Survived', palette='Set2')


# ***Sex***

# In[17]:


sns.countplot(train.Sex, hue=train.Survived)


# From `Title` and `Sex`, We can clearly see that `Female` passengers had **high chances of survival** than `Male`.

# **We can observe that Title and Sex are quite similar and highly correlate. This wouldn't be useful.
# We ll drop title.**

# In[18]:


train['isMale'] = train.Sex=='male'
test['isMale'] = test.Sex=='male'


# ***Ticket***
# 
# Let's observe and see if we can create any feature from the tickets.

# In[19]:


Ticket = train.Ticket.str.strip().str[0]
Ticket_test = test.Ticket.str.strip().str[0]


# In[20]:


plt.figure(figsize=(14,6))

sns.countplot('Ticket',data=train.drop('Ticket',1).join(Ticket),hue='Survived', palette="Accent")


# The Passengers with Ticket starting with 3 had **low survival rate**. While Passengers with ticket starting with `1` had **high survival rate**.

# The features extracted from `ticket` seem to be **useful**.
# Let's append them to our dataframe.
# Don't worry, we will see which features to keep during **feature selection**.

# In[21]:


train = train.drop('Ticket',1).join(Ticket).rename(columns={'Ticket':'Ticket'})
test = test.drop('Ticket',1).join(Ticket_test).rename(columns={'Ticket':'Ticket'})


# In[22]:


train['Ticket_preferred']=train.Ticket.apply(lambda s: s in ['3','A','S','C','7','W','4'])
test['Ticket_preferred']=test.Ticket.apply(lambda s: s in ['3','A','S','C','7','W','4'])


# ***Cabin***

# In[23]:


print(("Total no of null values in Cabin column:",train.Cabin.isnull().sum()))


# It is a lot of `null` values. Let's try to create `features` from the available values and we ll see afterwards if they are useful.

# In[24]:


Cabin = train.Cabin.str[0]
Cabin_test = test.Cabin.str[0]
Cabin.value_counts()


# In[25]:


#let's append it to our dataset
train = train.drop('Cabin',1).join(Cabin)
test = test.drop('Cabin',1).join(Cabin_test)


# In[26]:


sns.countplot(train.Cabin, hue=train.Survived)


# We can see almost every `cabintype` has high survival rate.
# 
# This might be due to them being in `first` and `second`.

# In[27]:


train['Cabin_preferred']=train.Cabin.apply(lambda s: s in ['C','E','D','B','F'])
test['Cabin_preferred']=test.Cabin.apply(lambda s: s in ['C','E','D','B','F'])


# ***Embarked***

# In[28]:


sns.countplot(train.Embarked, hue=train.Survived, palette='seismic')


# Passengers who embarked from port `S` had low chances of survival 
# (might be because most of them belonged to 3rd Pclass).
# 
# Passengers embarked from port `C` had high chances of survival.

# In[29]:


print(('Total no of missing values in Embarked:',train.Embarked.isna().sum()))


# We can clearly see the **mode** of the Embarked column is `S`.
# 
# So, let's fill the missing values with `S`.

# In[30]:


train.Embarked.fillna('S',inplace=True)


# In[31]:


train['ifEmbarkedS'] = train.Embarked =='S'
test['ifEmbarkedS'] = test.Embarked =='S'


# Let's drop the original columns as we have extracted features from them.

# In[32]:


train.drop('Pclass',1, inplace = True)
test.drop('Pclass',1, inplace = True)

train.drop('Name',1, inplace = True)
test.drop('Name',1, inplace = True)

train.drop('Title',1, inplace = True)
test.drop('Title',1, inplace = True)

train.drop('Sex',1, inplace = True)
test.drop('Sex',1, inplace = True)

train = train.drop('Ticket',1)
test = test.drop('Ticket',1)

train = train.drop('Cabin',1)
test = test.drop('Cabin',1)

train = train.drop('Embarked',1)
test = test.drop('Embarked',1)


# ### ***Now, let's analyse the numerical features***

# In[33]:


#descriptive statistics of numerical data
train.drop(['Survived'],1).describe()


# ## Age

# In[34]:


print(('Total no of missing values in Age:',train.Age.isna().sum()))


# This is large number considering the size of our dataset. So we need to be careful while filling the missing values.
# 
# We generated a new feature - Title and Ticket_numtype. It should be useful in imputing these missing values in Age.

# In[35]:


train.loc[:,'Age'] = train.loc[:,'Age'].fillna(train.groupby(['isMale','Ticket_preferred'])['Age'].transform('median'))
train.loc[:,'Age'] = train.loc[:,'Age'].fillna(train['Age'].median())


# In[36]:


test.loc[:,'Age'] = test.loc[:,'Age'].fillna(test.groupby(['isMale','Ticket_preferred'])['Age'].transform('median'))
test.loc[:,'Age'] = test.loc[:,'Age'].fillna(test['Age'].median())


# In[37]:


sns.distplot(train[train.Survived==0].Age,color='r')
sns.distplot(train[train.Survived==1].Age,color='g')


# In[38]:


train[['isPclass3','Age']].groupby('isPclass3').mean()


# We can observe that **Children** had **high chances of survival** (they might have been given preference during rescue operation).
# 
# Whereas people around the age of `20 - 30` had very **low chance of survival** (The reason might be that many of them belonged to the third class as we can see the average age of 3rd Pclass))
# 
# Also `old people (age 60+)` too had **low survival rate.**

# While studying about this dataset from various notebooks, I came across creation of Family_size. I feel it would be a good idea to include it.
# 
# Let's create Family_size by adding SibSp and Parch.

# In[39]:


train['Family_size'] = train['SibSp'] + train['Parch'] + 1
test['Family_size'] = test['SibSp'] + test['Parch'] + 1


# In[40]:


sns.distplot(train[train.Survived==0].Family_size,color='r')
sns.distplot(train[train.Survived==1].Family_size,color='g')


# We can observe very **low survival rate** among passengers who were travelling **alone**(might be due to preference to family people in rescue operation).

# In[41]:


train['isAlone']= train.Family_size==1
test['isAlone']= test.Family_size==1


# In[42]:


#let's drop sibsp, parch and family_size as their work is done.
train.drop(['SibSp','Parch','Family_size'],1,inplace=True)
test.drop(['SibSp','Parch','Family_size'],1,inplace=True)


# ***Fare***

# In[43]:


sns.boxplot(train.Fare, hue=train.Survived)


# In[44]:


train[train.Fare>400].Fare = np.nan
train.loc[:,'Fare'] = train.loc[:,'Fare'].fillna(train.groupby(['isMale','Ticket_preferred'])['Fare'].transform('mean'))
test.loc[:,'Fare'] = test.loc[:,'Fare'].fillna(test.groupby(['isMale','Ticket_preferred'])['Fare'].transform('mean'))


# Let's Zoom in.

# In[45]:


fig, ax = plt.subplots(1,2, figsize=(16,6))
sns.distplot(train[train.Survived==0].Fare,color='r', bins=50, ax=ax[0])
sns.distplot(train[train.Survived==1].Fare,color='g', bins=100, ax=ax[0])
ax[0].axis(xmin=-10,xmax=55)

sns.distplot(train[train.Survived==0].Fare,color='r', bins=75, ax=ax[1])
sns.distplot(train[train.Survived==1].Fare,color='g', bins=100, ax=ax[1])
ax[1].axis(xmin=45,xmax=150)


# We can see that cheap fare passengers had a very low survival rate(obvious).

# In[46]:


#binning age
train=train.drop('Age',1).join(pd.cut(train.Age, list(range(0,81,10)), True, list(range(8)), ).astype('int64')).rename(columns={'Age':'Age_bin'})
test=test.drop('Age',1).join(pd.cut(test.Age, list(range(0,81,10)), True, list(range(8)), ).astype('int64')).rename(columns={'Age':'Age_bin'})


# In[47]:


n=9
sns.countplot(pd.qcut(train.Fare, n, list(range(n))).astype('int64'), hue=train.Survived)


# ### Binning Fare in train and test data using pd.qcut()

# In[48]:


#binning Fare
train.Fare, bins = pd.qcut(train.Fare, n, list(range(n)), True)
test.Fare = pd.cut(test.Fare, bins, True, list(range(n)))


# In[49]:


train.Fare = train.Fare.astype('int64')
test.Fare = test.Fare.fillna(method='bfill').astype('int64')


# Let's create some artificial features by combining the categorical columns.

# In[50]:


# Custom Label Encoder for handling unknown values
class LabelEncoderExt(object):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, data):
        self.label_encoder = self.label_encoder.fit(list(data) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_
        return self

    def transform(self, data):
        new_data = list(data)
        for unique_item in np.unique(data):
            if unique_item not in self.label_encoder.classes_:
                new_data = ['Unknown' if x==unique_item else x for x in new_data]
        return self.label_encoder.transform(new_data)


# In[51]:


# from itertools import combinations

# object_cols = train.select_dtypes("object").columns
# object_cols_test = test.select_dtypes("object").columns

# low_cardinality_cols = [col for col in object_cols if train[col].nunique() < 15]

# interactions = pd.DataFrame(index=train.index)
# interactions_test = pd.DataFrame(index=test.index)

# # Iterate through each pair of features, combine them into interaction features
# for features in combinations(low_cardinality_cols,2):
    
#     new_interaction = train[features[0]].map(str)+"_"+train[features[1]].map(str)
#     new_interaction_test = test[features[0]].map(str)+"_"+test[features[1]].map(str)
    
#     encoder = LabelEncoderExt()
#     encoder.fit(new_interaction)
#     interactions["_".join(features)] = encoder.transform(new_interaction)
#     interactions_test["_".join(features)] = encoder.transform(new_interaction_test)


# In[52]:


# train = train.join(interactions) #append to the dataset
# test = test.join(interactions_test) #append to the dataset


# In[53]:


train.info()


# In[54]:


test.info()


# Our process of feature engineering is complete.
# 
# Now let's complete our preprocessing by converting categorical data into numerical data.

# In[55]:


print((train.isna().sum(), test.isna().sum()))


# In[56]:


X_train = train.drop('Survived',1)
y_train = train.Survived
X_test = test


# In[57]:


seed = 42


# In[58]:


sampler = SMOTE(random_state = seed)
X_train, y_train = sampler.fit_resample(X_train, y_train)


# In[59]:


y_train.value_counts()


# In[60]:


print((X_train.shape,X_test.shape))


# In[61]:


# pcorr = X_train.corrwith(y_train)
# imp_corr_cols = pcorr[(pcorr>0.1) | (pcorr<-0.1)].index

# X_train = X_train[imp_corr_cols]
# X_test = X_test[imp_corr_cols]


# ## Model imports

# In[62]:


from numpy import mean
from numpy import std

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV


# In[63]:


models = {
    'LR':make_pipeline(StandardScaler(),LogisticRegression(random_state=seed)),
    'SVC':make_pipeline(StandardScaler(),SVC(random_state=seed)),
    'AB':AdaBoostClassifier(random_state=seed),
    'ET':ExtraTreesClassifier(random_state=seed),
    'GB':GradientBoostingClassifier(random_state=seed),
    'RF':RandomForestClassifier(random_state=seed),
    'XGB':XGBClassifier(random_state=seed),
    'LGBM':LGBMClassifier(random_state=seed)
    }


# In[64]:


# evaluate a give model using cross-validation
def evaluate_model(model):
    cv = StratifiedKFold(shuffle=True, random_state=seed)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[65]:


# evaluate the models and store results
results, names = list(), list()
for name, model in list(models.items()):
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print(('*%s %.3f (%.3f)' % (name, mean(scores), std(scores))))


# In[66]:


plt.boxplot(results, labels=names, showmeans=True)
plt.show()


# ***Feature Selection using mean of feature importance of different models***

# In[67]:


feat_imp=[]
for name, model in list(models.items()):
    model.fit(X_train, y_train)
    if name not in ['LR', 'SVC', 'KNN']: #since they do not have feature importance
        feat_imp.append(pd.Series(model.feature_importances_, index=X_train.columns))


# In[68]:


feat_imp[-1]=feat_imp[-1].apply(lambda x: x/1000)
avg_feat_imp = pd.DataFrame(feat_imp).mean()
plt.figure(figsize=(16,6))
plt.xticks(rotation=90)
plt.xlabel('Average Feature Importance')
plt.plot(avg_feat_imp.sort_values(ascending=False))


# In[69]:


# impcols = avg_feat_imp.sort_values(ascending=False).index[:15]
# X_train = X_train[impcols]
# X_test = X_test[impcols]


# In[70]:


models = {
    'SVC':make_pipeline(StandardScaler(),SVC(random_state=seed)),
    'AB':AdaBoostClassifier(random_state=seed),
    'ET':ExtraTreesClassifier(n_jobs=-1, random_state=seed),
    'GB':GradientBoostingClassifier(random_state=seed),
    'RF':RandomForestClassifier(n_jobs=-1, random_state=seed),
    'XGB':XGBClassifier(n_jobs=-1, random_state=seed),
    'LGBM':LGBMClassifier(n_jobs=-1, random_state=seed)
    }


# In[71]:


sns.heatmap(X_train.join(y_train).corr().apply(abs), cmap = 'bwr')


# In[72]:


for name, model in list(models.items()):
    print((name,'parameters:'))
    print((model.get_params()))
    print(('='*140))


# In[73]:


params = {
    'SVC':{'svc__gamma':[0.01,0.02,0.05,0.08,0.1], 'svc__C':list(range(1,8))},
    
    'AB':{'learning_rate': [0.05, 0.1, 0.2, 0.5], 'n_estimators': list(range(50,501,100))},
    
    'ET':{'max_depth':[5,8,10,12], 'min_samples_split': [5,8,10,12],
          'n_estimators': [500,1000,1500,2000]},
    
    'GB':{'learning_rate': [0.1, 0.2, 0.5], 'max_depth':[3,5,8,10],
          'min_samples_split': [5,8,10,12], 'n_estimators': [50,100,200,500],
          'subsample':[0.5,0.7,0.9]},
    
    'RF':{'max_depth':[3,5,10,12,15], 'n_estimators': [50,100,500,1000],
          'min_samples_split': [4,8,10]},
    
    'XGB':{'max_depth':list(range(3,10,2)), 'n_estimators': list(range(50,201,50)),
           'learning_rate': [0.05, 0.08, 0.1, 0.15], 'subsample':[0.5,0.7,0.9]},
    
    'LGBM':{'max_depth':list(range(3,10,2)), 'n_estimators': list(range(50,201,50)),
            'learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2],'subsample':[0.5,0.7,0.9],
           'num_leaves': list(range(15,51,10))}
}


# In[74]:


# evaluate the models and store results
best_params = params
names= list()
for name, param_grid, model in zip(list(params.keys()), list(params.values()), list(models.values())):
    gscv = GridSearchCV(model, param_grid, n_jobs=-1, verbose=3, cv=4)
    gscv.fit(X_train,y_train)
    names.append(name)
    best_params[name] = gscv.best_params_
    print(name)
    print(("best score:",gscv.best_score_))
    print(("best params:",gscv.best_params_))


# In[75]:


base_models = [
    ('SVC',make_pipeline(StandardScaler(),SVC(random_state=seed))),
    ('AB',AdaBoostClassifier(random_state=seed)),
    ('ET',ExtraTreesClassifier(random_state=seed)),
    ('GB',GradientBoostingClassifier(random_state=seed)),
    ('RF',RandomForestClassifier(random_state=seed)),
    ('XGB',XGBClassifier(random_state=seed)),
    ('LGBM',LGBMClassifier(random_state=seed))
]


# In[76]:


for model, param in zip(base_models,list(best_params.values())):
    model[1].set_params(**param)


# In[77]:


clf = StackingClassifier(estimators=base_models)


# In[78]:


score = evaluate_model(clf)


# In[79]:


print(score)
print((mean(score)))


# In[80]:


clf.fit(X_train,y_train)


# In[81]:


y_preds = clf.predict(X_test)


# In[82]:


Submission = pd.DataFrame({ 'PassengerId': PID,'Survived': y_preds })
Submission.to_csv('Submission.csv', index = False)


# In[ ]:




