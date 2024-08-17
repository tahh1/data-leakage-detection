#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[2]:


from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[3]:


def LabelEncoding(all_data,data):
    le = LabelEncoder()
    le.fit(all_data)
    return le.transform(data)


# # 1. Make Feature Map

# In[4]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
data_all = pd.concat([train, test],axis=0,sort=True).reset_index().drop(['index'],axis=1)

train.name = 'Train data'
test.name = 'Test data'
data_all.name = 'All data'


# In[5]:


print((train.name))
train.info()


# In[6]:


print((test.name))
test.info()


# ## 1.1. One-Hot Encoding

# Transformed into One-Hot representation.

# ### 1.1.1. Sex

# In[7]:


train = train.assign(Female=pd.get_dummies(train['Sex'])['female'],
                     Male=pd.get_dummies(train['Sex'])['male'])

test = test.assign(Female=pd.get_dummies(test['Sex'])['female'],
                     Male=pd.get_dummies(test['Sex'])['male'])

train[['Male','Female']].sample(5)


# ### 1.1.2. Pclass

# Pclass 1:Upper Class  
# Pclass 2:Middle Class  
# Pclass 3:Lower Class

# In[8]:


#Pclass is transformed  into one-hot
train_pclass = pd.get_dummies(train['Pclass'])
test_pclass = pd.get_dummies(test['Pclass'])

train = train.assign(Pclass_1 = train_pclass[1],
                     Pclass_2 = train_pclass[2],
                     Pclass_3 = train_pclass[3],
                    )
test = test.assign(Pclass_1 = test_pclass[1],
                   Pclass_2 = test_pclass[2],
                   Pclass_3 = test_pclass[3],
                  )
train[['Pclass_1','Pclass_2','Pclass_3']].sample(5)


# ## 1.1.3. Tittle

# Tittle is extracted from Name.

# In[9]:


from keras.preprocessing import text 

tokenizer = text.Tokenizer()

train_texts = train['Name'].values
test_texts = test['Name'].values

tokenizer.fit_on_texts(train_texts)
train_list_tokenized = tokenizer.texts_to_sequences(train_texts)

train_mstr = np.zeros(train.shape[0]).astype(int)
train_mr = np.zeros(train.shape[0]).astype(int)
train_miss = np.zeros(train.shape[0]).astype(int)
train_mrs = np.zeros(train.shape[0]).astype(int)

for i,name in enumerate(train_list_tokenized):
    #Master:6,Mr:1,Miss:2,Mrs:3
    if 6 in name: 
        train_mstr[i] = 1
    elif 3 in name:
        train_mr[i] = 1
    elif 2 in name:
        train_miss[i] = 1
    elif 1 in name:
        train_mrs[i] = 1

tokenizer.fit_on_texts(test_texts)
test_list_tokenized = tokenizer.texts_to_sequences(test_texts)

test_mstr = np.zeros(test.shape[0]).astype(int)
test_mr = np.zeros(test.shape[0]).astype(int)
test_miss = np.zeros(test.shape[0]).astype(int)
test_mrs = np.zeros(test.shape[0]).astype(int)
for i,name in enumerate(test_list_tokenized):
    #Master:6,Mr:1,Miss:2,Mrs:3
    if 6 in name: 
        test_mstr[i] = 1
    elif 3 in name:
        test_mr[i] = 1
    elif 2 in name:
        test_miss[i] = 1
    elif 1 in name:
        test_mrs[i] = 1
        
train = train.assign(Mstr=train_mstr,Mr=train_mr,Miss=train_miss,Mrs=train_mrs)
test = test.assign(Mstr=test_mstr,Mr=test_mr,Miss=test_miss,Mrs=test_mrs)

train[['Mstr','Mr','Miss','Mrs']].sample(5)


# ### 1.1.4. Embarked

# In[10]:


train['Embarked'] = train['Embarked'].fillna('S') # S is the most frequency
train_embarked = pd.get_dummies(train['Embarked'])
test_embarked = pd.get_dummies(test['Embarked'])

train = train.assign(Embarked_S = train_embarked['S'],
                     Embarked_C = train_embarked['C'],
                     Embarked_Q = train_embarked['Q'])

test = test.assign(Embarked_S = test_embarked['S'],
                   Embarked_C = test_embarked['C'],
                   Embarked_Q = test_embarked['Q'])
train[['Embarked_S','Embarked_C','Embarked_Q']].sample(5)


# ## 1.2. Label Encoding

# ### 1.2.1. Fare

# In[11]:


print((data_all.groupby(['Pclass','Embarked'])['Fare'].mean()))
data_all[data_all['Fare'].isnull()]


# In[12]:


#Null of fare is predicted low cost because of this person is in  Lower_class and Embarke S
data_all['Fare'] = data_all['Fare'].fillna(14.43)


# In[13]:


fare_categories = pd.qcut(data_all['Fare'], 10,labels=False)
train['Fare'] = fare_categories[:train.shape[0]].values
test['Fare'] = fare_categories[train.shape[0]:].values
train[['Fare']].sample(5)


# ### 1.2.2. Age

# In[14]:


age_categories = pd.qcut(data_all['Age'], 5,labels=False)
train['Age'] = age_categories[:train.shape[0]].fillna(-1).values #null is filled -1
test['Age']  = age_categories[train.shape[0]:].fillna(-1).values #null is filled -1
test[['Age']].sample(5)


# ### 1.2.3. Cabin

# In[15]:


data_all['Cabin'] = data_all['Cabin'].fillna('Z')
Cabin_all = [i[0] for i in data_all['Cabin']]

train_cabin = [i for i in Cabin_all[:train.shape[0]]]
test_cabin = [i for i in Cabin_all[train.shape[0]:]]

train['Cabin'] = LabelEncoding(Cabin_all,train_cabin)
test['Cabin'] = LabelEncoding(Cabin_all,test_cabin)


# ## 1.3. Family Size

# In[16]:


#Family_sizes
family_size_group1_train = np.zeros(train.shape[0]).astype(int)
family_size_group2_train = np.zeros(train.shape[0]).astype(int)
family_size_group3_train = np.zeros(train.shape[0]).astype(int)
alone_train = np.zeros(train.shape[0]).astype(int)

family_size_group1_test = np.zeros(test.shape[0]).astype(int)
family_size_group2_test = np.zeros(test.shape[0]).astype(int)
family_size_group3_test = np.zeros(test.shape[0]).astype(int)
alone_test = np.zeros(test.shape[0]).astype(int)

family_size_all_train = train['SibSp'] +train['Parch']
family_size_all_test = test['SibSp'] +test['Parch']

for i,f_size in enumerate(family_size_all_train):
    if f_size == 1 or f_size == 2:
        family_size_group1_train[i] = 1
    elif f_size == 3 or f_size == 4 or f_size == 5:
        family_size_group2_train[i] = 1
    elif f_size == 0:
        alone_train[i] = 1
    else:
        family_size_group3_train[i] = 1

for i,f_size in enumerate(family_size_all_test):
    if f_size == 1 or f_size == 2:
        family_size_group1_test[i] = 1
    elif f_size == 3 or f_size == 4 or f_size == 5:
        family_size_group2_test[i] = 1
    elif f_size == 0:
        alone_test[i] = 1
    else:
        family_size_group3_test[i] = 1
        
train = train.assign(Alone = alone_train,
                     Family_group_1 = family_size_group1_train,
                     Family_group_2 = family_size_group2_train,
                     Family_group_3 = family_size_group3_train)

test = test.assign(Alone = alone_test,
                   Family_group_1 = family_size_group1_test,
                   Family_group_2 = family_size_group1_test,
                   Family_group_3 = family_size_group1_test)


# ## 1.4. Ticket Frequency

# In[17]:


test['Ticket'] = data_all.groupby('Ticket')['Ticket'].transform('count')[train.shape[0]:].values
train['Ticket'] = data_all.groupby('Ticket')['Ticket'].transform('count')[:train.shape[0]].values


# Drop unnecessary features

# In[18]:


train = train.drop(['Sex','Name','Embarked','Pclass'],axis=1)
test = test.drop(['Sex','Name','Embarked','Pclass'],axis=1)


# ## 1.5. Predict Age

# We can predict the null of age in this dataset. I utilized Random Forest Classifier(RFC).

# In[19]:


#predict_age
from sklearn.ensemble import RandomForestClassifier as RFC

new_data_all = pd.concat([train.drop(['Survived'],axis=1),test],axis=0).reset_index().drop(['index'],axis=1)
not_null_age_index = [i for i,flag in enumerate(new_data_all['Age']) if flag != -1]
null_age_index = [i for i,flag in enumerate(new_data_all['Age']) if flag == -1]

train_null_age_index = null_age_index[:177]
test_null_age_index = [i-train.shape[0] for i in null_age_index[177:]]

train_age_data_x = new_data_all.iloc[not_null_age_index].drop(['PassengerId','Age'],axis=1)
train_age_data_y = new_data_all.iloc[not_null_age_index]['Age']

test_age_data = new_data_all.iloc[null_age_index].drop(['PassengerId','Age'],axis=1)


# In[20]:


accuracy = []
pred_result = np.ones(train_age_data_x.shape[0])
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=71)
for tr_idx, va_idx in kf.split(train_age_data_x,train_age_data_y):
    tr_x,va_x = train_age_data_x.iloc[tr_idx], train_age_data_x.iloc[va_idx]
    tr_y,va_y = train_age_data_y.iloc[tr_idx], train_age_data_y.iloc[va_idx]
    model = RFC(criterion='gini',
                n_estimators=100,
                max_depth=8,
                min_samples_split=7,
                min_samples_leaf=7,
                max_features='auto',
                oob_score=True,
                random_state=71,
                n_jobs=-1,
                verbose=0) 
    model.fit(tr_x,tr_y)

    pred=model.predict(va_x)
    accuracy.append((sum(pred==va_y))/len(va_y))
print(('Varidation accuracy:',accuracy))
print(('Average of accuracy:',np.mean(accuracy)))


# In[21]:


model.fit(train_age_data_x,train_age_data_y)

age_pred=model.predict(test_age_data)


# In[22]:


train['Age'].iloc[train_null_age_index] = age_pred[:177]
test['Age'].iloc[test_null_age_index] = age_pred[177:]


# ## 1.6. Survived Rate

# In[23]:


New_data_all = pd.concat([train,test],axis = 0,sort=True).reset_index().drop(['index','PassengerId'],axis=1)
plt.subplots(figsize=(10, 7)) 

survived_rate = New_data_all.groupby(['Ticket','Alone','Family_group_1','Family_group_2','Family_group_3'])['Survived'].transform('mean')
survived_rate2 = New_data_all.groupby(['Parch','SibSp','Male','Female'])['Survived'].transform('mean')

New_data_all = New_data_all.assign(Survived_rate = survived_rate)
New_data_all = New_data_all.assign(Survived_rate2 = survived_rate2)

sns.heatmap(New_data_all.corr(), vmax=1, vmin=-1, center=0)


# In[24]:


New_all_data = pd.concat([train,test],axis=0,sort=True).reset_index().drop(['index'],axis=1)

New_train = train.assign(Survived_rate = New_data_all['Survived_rate'][:train.shape[0]].values)
New_train = New_train.assign(Survived_rate2 = New_data_all['Survived_rate2'][:train.shape[0]].values)

New_test = test.assign(Survived_rate = New_data_all['Survived_rate'][train.shape[0]:].values)
New_test = New_test.assign(Survived_rate2 = New_data_all['Survived_rate2'][train.shape[0]:].values)

plt.subplots(figsize=(10, 7)) 
sns.heatmap(New_train.corr(), vmax=1, vmin=-1, center=0)
New_test = New_test.fillna(0.5)


# # 2. Model

# In[25]:


train_x = New_train.drop(['PassengerId','Survived'],axis=1)
train_x = train_x.astype('float32')
train_y = New_train['Survived']

test_x = New_test.drop(['PassengerId'],axis=1)
test_x


# ## 2.1. Random Forest Classifier

# In[26]:


scores_accuracy = []
pred_result = np.ones(train.shape[0])
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=71)
for tr_idx, va_idx in kf.split(train_x,train_y):
    tr_x,va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y,va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = RFC(criterion='gini',
                n_estimators=100,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=10,
                max_features='auto',
                oob_score=True,
                random_state=71,
                n_jobs=-1,
                verbose=0) 
    model.fit(tr_x,tr_y)

    pred=model.predict(va_x)
    accuracy = accuracy_score(va_y, pred>0.5)
    scores_accuracy.append(accuracy)
    
print(('Varidation accuracy:',scores_accuracy))
print(('Average of accuracy:',np.mean(scores_accuracy)))


# In[27]:


model.fit(train_x,train_y)
RFC_pred=model.predict(test_x)


# ## 2.2. Support Vector Machine

# In[28]:


from sklearn.svm import SVC
scores_accuracy = []
pred_result = np.ones(train.shape[0])
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=71)
for tr_idx, va_idx in kf.split(train_x,train_y):
    tr_x,va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y,va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = SVC(kernel='rbf',C = 100,gamma = 0.001, random_state=71)
    model.fit(tr_x,tr_y)

    pred=model.predict(va_x)
    accuracy = accuracy_score(va_y, pred>0.5)
    scores_accuracy.append(accuracy)
    
print(('Varidation accuracy:',scores_accuracy))
print(('Average of accuracy:',np.mean(scores_accuracy)))


# In[29]:


model.fit(train_x,train_y)
SVC_pred=model.predict(test_x)


# ## 2.3. Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
scores_accuracy = []
pred_result = np.ones(train.shape[0])
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=71)
for tr_idx, va_idx in kf.split(train_x,train_y):
    tr_x,va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y,va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    model = LogisticRegression(random_state=71,n_jobs=-1,verbose=0)
    model.fit(tr_x,tr_y)

    pred=model.predict(va_x)
    accuracy = accuracy_score(va_y, pred>0.5)
    scores_accuracy.append(accuracy)
    
print(('Varidation accuracy:',scores_accuracy))
print(('Average of accuracy:',np.mean(scores_accuracy)))


# In[31]:


model.fit(train_x,train_y)
LR_pred=model.predict(test_x)


# ## 2.4. XGboost

# In[32]:


from xgboost import XGBClassifier

scores_accuracy = []
pred_result = np.ones(train.shape[0])
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=71)
for tr_idx, va_idx in kf.split(train_x,train_y):
    tr_x,va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y,va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
#     model = XGBClassifier(n_estimators=20, random_state=71)
    model = XGBClassifier(criterion='gini',
                n_estimators=20,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=4,
                max_features='auto',
                oob_score=True,
                random_state=71,
                n_jobs=-1,
                verbose=0)     
    model.fit(tr_x,tr_y)

    pred=model.predict(va_x)
    accuracy = accuracy_score(va_y, pred>0.5)

    scores_accuracy.append(accuracy)

print(('Varidation accuracy:',scores_accuracy))
print(('Average of accuracy:',np.mean(scores_accuracy)))


# In[33]:


model.fit(train_x,train_y)
XGboost_pred=model.predict(test_x)


# # 3. Make submissions

# The best score is submission_3.(score:0.80382)

# ## 3.1. Using best performance model(SVC model)

# In[34]:


prediction = np.empty(test.shape[0])
for i in range(test.shape[0]):
    prediction[i] = 1 if SVC_pred[i] >0.5 else 0


# In[35]:


data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':prediction.astype(int)
})
data_to_submit.to_csv('submission_1.csv', index = False)


# ## 3.2. Using good performance model(RFC model)

# In[36]:


prediction = np.empty(test.shape[0])
for i in range(test.shape[0]):
    prediction[i] = 1 if RFC_pred[i] >0.5 else 0


# In[37]:


data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':prediction.astype(int)
})
data_to_submit.to_csv('submission_2.csv', index = False)


# ## 3.3. Ensemble good performance model

# In[38]:


prediction = np.empty(test.shape[0])
for i in range(test.shape[0]):
    prediction[i] = 1 if (SVC_pred[i]+RFC_pred[i]+XGboost_pred[i]+LR_pred[i])/4 >0.5 else 0


# In[39]:


data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':prediction.astype(int)
})
data_to_submit.to_csv('submission_3.csv', index = False)

