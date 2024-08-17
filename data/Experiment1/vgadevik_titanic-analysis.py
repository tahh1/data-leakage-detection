#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Analysis

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train_set =  pd.read_csv('../input/train.csv')
train_set.head()
# test_set.head()


# In[3]:


train_set.dtypes


# In[4]:


train_set.corr()


# In[5]:


train_set.describe()


# In[6]:


train_set.info()


# This means <b>Age</b>, <b>Cabin</b> and <b>Embarked</b> have some null values.

# In[7]:


train_set.isnull().sum()


# <ul>**Analysis** : <p>
#     <li><b>Age</b>, <b>Cabin</b> and <b>Embarked</b> have the above number of null values.</li> 
#         <li><b>Cabin</b> has almost 77% of the values to be null and thus there is no point in imputing them. So we can remove <b>Cabin</b>.</li>
#     <li>We neeed to impute the missing values of <b>Age</b> and <b>Embarked</b>.</li>
#     <li><b>Ticket</b> and <b>PassengerId</b> intuitively don't contribute to the target <b>Survived</b>. Hence they can be removed.</li>
#     <li>We can retrieve special attributes like designition or gender (basically <b>Title</b>) from the titles given as part of <b>Names</b>. So we can extract the title from the name and create a column like <b>Title</b> to put the title.</li>
#     </ul>

# ### **Survivval status of the Passengers**

# In[8]:


plt_survived = train_set.Survived.value_counts().plot('bar')
plt_survived.set_xlabel("Survived")
plt_survived.set_ylabel("No of passengers")
plt_survived.set_title("Survivval status of Passengers"+" ("+"Not survived - 0 and Survived - 1"+")")
for p in plt_survived.patches:
    plt_survived.annotate(str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.005))


# ### **Survival probability based on the PClass**

# In[9]:


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(221)
ax1 = fig.add_subplot(222)

max_value = train_set.Pclass.value_counts().values[0]
max_value = (int(max_value/100)+1)*100
plt_pclass = train_set.Pclass.value_counts().sort_index().plot(kind = 'bar', ax=ax)
plt_pclass.set_xlabel("Pclass")
plt_pclass.set_ylabel("No of Passengers")
plt_pclass.set_title("Total Passengers in each PClass Category")

for p in plt_pclass.patches:
    plt_pclass.annotate(str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.005))

# import matplotlib.pyplot as plt


survived_df = train_set.loc[train_set.Survived==1]
survived_df.head()


plt_pclass_survived = survived_df.Pclass.value_counts().sort_index().plot(kind = 'bar', ax=ax1, ylim=[0, max_value])
plt_pclass_survived.set_xlabel("Pclass")
plt_pclass_survived.set_ylabel("No of Passengers")
plt_pclass_survived.set_title("Survived Passengers in each PClass Category")

for p in plt_pclass_survived.patches:
    plt_pclass_survived.annotate(str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.005))


# In[10]:


pclass_count = list(train_set['Pclass'].value_counts().sort_index())
survived_df = train_set[train_set.Survived==1]
survived_count = list(survived_df.Pclass.value_counts().sort_index())
index = sorted(train_set.Pclass.unique())
df = pd.DataFrame({'Total passengers':pclass_count,'Survived passengers':survived_count},index=index,columns=['Total passengers','Survived passengers'])
ax = df.plot.bar(rot=0)
ax.set_xlabel("PClass")
ax.set_ylabel("Passengers count")
ax.set_title("PClass wise Survival status of passengers")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.025, p.get_height() * 1.005))


# **Analysis** : About 63% of PClass 1, 47% of PClass 2 and 24% of PClass 3 are survived.

# ### **Survival probability based on the Embarked**

# In[11]:


plt = train_set[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')
for p in plt.patches:
    plt.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.025, p.get_height() * 1.005))
plt.set_title("Survival probability based on the Embarked")


# **Analysis** :  Thus the survival probability is in the order: **C>Q>S**

# > ### **Survival probability based on the Gender**

# In[12]:


plt = train_set[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot('bar')
for p in plt.patches:
    plt.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.025, p.get_height() * 1.005))
plt.set_title("Survival probability based on the Gender")


# **Analysis** : Thus, Females have higher Survival probability.

# ### **Survival probability based on the No of Siblings or Spouses(SibSp)**

# In[13]:


plt = train_set.SibSp.value_counts().sort_index().plot('bar')
plt.set_xlabel('SibSp')
plt.set_ylabel('Passenger count')
for p in plt.patches:
    plt.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.025, p.get_height() * 1.005))


# In[14]:


plt = train_set[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot('bar')
plt.set_xlabel('SibSp')
plt.set_ylabel('Survival Probability')
for p in plt.patches:
    plt.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.025, p.get_height() * 1.005))
plt.set_title("Survival probability based on the SibSp")


# **Analysis** : 
# 
# * As we can see, people with 1 or 2 siblings or spouses have higher probability of suriving.
# 
# * Also, the survival probabilities for the people with 4-8 siblings or spouses are bitterly low.

# ### **Extracting 'Title' from the name as a new feature**

# In[15]:


titles_list = train_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False).tolist()
# train_set.drop(columns=['Name'], inplace=True)
train_set = train_set.drop(columns=['Name'])
print((train_set.columns))
train_set['Title'] = titles_list
ax = train_set.Title.value_counts().plot('bar')
ax.set_title("Count of each Titles")
ax.set_ylabel("Count")
ax.set_xlabel("Title")


# In[16]:


train_set['Title'] = train_set['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_set['Title'] = train_set['Title'].replace('Ms', 'Miss')
train_set['Title'] = train_set['Title'].replace('Mme', 'Mrs')
train_set['Title'] = train_set['Title'].replace('Mlle', 'Miss')
ax = train_set.Title.value_counts().plot('bar')
ax.set_title("Count of each Titles")
ax.set_ylabel("Count")
ax.set_xlabel("Title")


# ### **Survival probability based on the Title**

# In[17]:


plt = train_set[['Title', 'Survived']].groupby('Title').mean().Survived.plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Survival Probability')
plt.set_title("Survival probability based on the Title")
for p in plt.patches:
    plt.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.025, p.get_height() * 1.005))


# **Analysis** : 
# In accordance with the higher survival probability of females, the survival probability of people with 'Mrs' and 'Miss' are higher.

# ### **Creating Training Set**

# In[18]:


X_train = train_set[train_set.columns.difference(['Survived'])]
X_train.head()


# In[19]:


Y_train = train_set['Survived']
Y_train.head()


# In[20]:


c = X_train.Cabin.value_counts()
print(("Values: ",c.size))
cabin_Nan = X_train.loc[ (pd.isna(X_train['Cabin'])) , 'Cabin' ].shape[0]
print(("Nan :",cabin_Nan))
print((X_train.shape[0]))

# thus 'Cabin' may not be an important feature


# In[21]:


c = X_train.Ticket.value_counts()
print(("Values: ",c.size))
ticket_Nan = X_train.loc[ (pd.isna(X_train['Ticket'])) , 'Ticket' ].shape[0]
print(("Nan :",ticket_Nan))
print(("Data: ", X_train.shape[0]))
print(("Unique: ",len(X_train.Ticket.unique())))

# thus 'Ticket' may not be an important feature


# In[22]:


X_train = train_set[train_set.columns.difference(['Cabin','Ticket'])]
# X_train['Embarked']= X_train['Embarked'].astype("category").cat.codes
# X_train['Sex']= X_train['Sex'].astype("category").cat.codes

X_train.head()


# In[23]:


X_train.corr()


# In[24]:


X_train = X_train[X_train.columns.difference(['Survived','PassengerId','Ticket'])]
X_train.head()


# ##### **Converting 'Sex' and 'Embarked' from categorical to numerical data types**

# In[25]:


X_train['Sex'] = X_train['Sex'].map({'male':0, 'female':1})
X_train['Embarked'] = X_train['Embarked'].map({'C':0, 'Q':1, 'S':2})
X_train.head()


# In[26]:


Y_train.corr(X_train['Pclass'])


# In[27]:


# X_train['Title']= X_train['Title'].astype("category").cat.codes
X_train['Title'] = X_train['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
X_train.head()


# ##### **Filling Null Values**

# In[28]:


X_train.isna().any()


# In[29]:


X_train.Age.isna().sum()


# In[30]:


X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_train.isna().sum()


# In[31]:


X_train['Embarked'] = X_train['Embarked'].fillna(X_train['Embarked'].max())
X_train.isna().sum()


# ## Preprocessing Test Data

# In[32]:


test_set =  pd.read_csv('../input/test.csv')


# In[33]:


test_set.isnull().sum()


# **Analysis** : As done for the train data, first we shall drop <b>PassengerId</b>, <b>Ticket</b> and <b>Cabin</b>.

# In[34]:


test_set = test_set.drop(columns=['Ticket', 'PassengerId', 'Cabin'])


# In[35]:


test_set['Title'] = test_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_set = test_set.drop(columns='Name')

test_set['Title'] = test_set['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_set['Title'] = test_set['Title'].replace('Ms', 'Miss')
test_set['Title'] = test_set['Title'].replace('Mme', 'Mrs')
test_set['Title'] = test_set['Title'].replace('Mlle', 'Miss')

test_set['Title'] = test_set['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
test_set.head()


# **Analysis** : Now, let us move on to imputing the missing values.

# In[36]:


test_set['Age'] = train_set['Age'].fillna(test_set['Age'].mean())
test_set.head()


# In[37]:


test_set.isna().sum()


# In[38]:


test_set['Fare'] = test_set.Fare.fillna(train_set.Fare.mean())


# In[39]:


row = test_set[test_set['Title'].isnull()].index
sex = test_set.iloc[row].Sex.values[0]
if sex == 'female':
    test_set['Title'].iloc[row] = 3
else:
    test_set['Title'].iloc[row] = 2
test_set["Title"] = test_set['Title'].astype('int64')
test_set.isna().sum()


# In[40]:


test_set.head()


# **Analysis** : Encoding <b>Sex</b> and <b>Embarked</b> with numerical values:

# In[41]:


test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1})
test_set['Embarked'] = test_set['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_set.head()


# In[42]:


test_cols = sorted(test_set.columns.tolist())
print(test_cols)
train_cols = sorted(X_train.columns.tolist())
print(train_cols)


# In[43]:


X_train.isna().sum()


# In[44]:


from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam


# In[45]:


X = X_train.values
Y = Y_train.values
X.shape


# In[46]:


model = Sequential()
model.add(Dense(15, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
sgd = Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[47]:


history = model.fit(X, Y, validation_split=0.1, epochs=88, batch_size=10)


# In[48]:


print((list(history.history.keys())))
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ** Analysis: **
# - As the epochs are increasing, the training accuracy becomes constant after 40 of epochs.
# - Accuracy is inversely proportional to loss.

# In[49]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

X, x, Y, y = train_test_split(X_train, Y_train, test_size=0.1, random_state=1) # 90% training and 10% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X,Y)

#Predict the response for test dataset
y_pred = clf.predict(x)

print(("Accuracy:",metrics.accuracy_score(y, y_pred)))


# In[50]:





# In[50]:





# In[50]:





# In[50]:




