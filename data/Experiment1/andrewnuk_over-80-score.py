#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard data analysis modules

import pandas as pd
import numpy as np
from IPython.display import display

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = None


# In[2]:


# check on the panda version and its dependencies
# i run this from time to time to ensure all is up to date
pd.__version__
#pd.show_versions()


# In[3]:


# i use these to input the relevant file names
# which i downloaded earlier and sit in the same directory as this
file_train = '/kaggle/input/titanic/train.csv'
file_test = '/kaggle/input/titanic/test.csv'
file_genderSubmission = '/kaggle/input/titanic/gender_submission.csv'


# In[4]:


# which makes this a standard cell
# use df for the training data, we will be spliting it up into training and validaing later anyway
# df is the training data, df_test is the test data
df = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
df_genderSubmission = pd.read_csv(file_genderSubmission)


# In[5]:


df1 = pd.read_csv(file_train)
df_test1 = pd.read_csv(file_test)


# In[6]:


#
# initial EDA
#


# In[7]:


df.shape, df_test.shape


# In[8]:


# comparing the columns in the training and test data to see if any differences
# seems in this case the only difference is the 'Survived' column in the training data
# as expected

set(list(df.columns)) - set(list(df_test.columns))


# In[9]:


df.head(10)


# In[10]:


df.describe()


# In[11]:


# so, most of the data appears to be categorical
# on the Fare data, there are some at 0 which is odd, will look into that later


# In[12]:


df.columns, df_test.columns


# In[13]:


#
# going through each column seperately
# i will deal with missing data as i go
#


# In[14]:


# missing values in the training data

data = {'name': df.isnull().sum().index,
       'count_training': list(df.isnull().sum())}

df_null = pd.DataFrame (data)
df_null['pct_training'] = df_null['count_training']/len(df)
df_null = df_null[df_null['name'] != 'Survived'] # drop the survived row
df_null['count_test'] = list(df_test.isnull().sum()) # columns in test and training are in the same order
df_null['pct_test'] = df_null['count_test']/len(df)
df_null = df_null[df_null['count_training'] + df_null['count_test'] != 0] # drop all rows with no missing data
df_null


# In[15]:


# Age data is missing in both training and test data, 20% missing in training
# 1 item of fare data is missing in test data
# 77% of cabin data is missing in training and 37% in test data
# 2 items of Embarked data are missing in training


# In[16]:


# i will deal with the Fare and Embarked missing data now given it is 1 & 2 points only

# looking at Fare first, even though i know there is something odd going on in Fares
# given there are zero value fares
# the missing Fare is in the test data

df_test[df_test['Fare'].isnull()]


# In[17]:


### single man of age 61 in third class.

# lets see if there are any other Storey's on baord - answer is no

df[df['Name'].str.contains('Storey')]['Name'],df_test[df_test['Name'].str.contains('Storey')]['Name']


# In[18]:


# see if there are any other names on the ticket 3701 - answer is no

df[df['Ticket'] == '3701']['Name'], df_test[df_test['Ticket'] == '3701']['Name']


# In[19]:


# so, lets take the average Fare of a male over 30, single, ticket count 1, embarked at S, third class

mean_fare = df_test[(df_test['Pclass'] == 3) & (df_test['Embarked'] == 'S') & (df_test['Sex'] == 'male') & 
   (df_test['SibSp'] == 0) & (df_test['Parch'] == 0) & (df_test['Age'] > 30) & 
        (df_test.groupby('Ticket')['Ticket'].transform('count') == 1)]['Fare'].mean()

df_test['Fare'] = df_test['Fare'].fillna(mean_fare)

mean_fare


# In[20]:


# looking at Embarked
# the missing data is in the training data
# techncially i could ignore it given the test data is not affected and it is only 2 data points

df[df['Embarked'].isnull()]


# In[21]:


# this looks like a woman and her maid

# lets see if we can find anyone else by the name of Stone or Icard - answer is no
# using lists to print out entire name

list1 = list(df[df['Name'].str.contains('Stone') | df['Name'].str.contains('Icard')]['Name'])
list2 = list(df_test[df_test['Name'].str.contains('Stone') | df_test['Name'].str.contains('Icard')]['Name'])

list1, list2


# In[22]:


# see if anyone else is on the same ticket - asnwer is no

df[df['Ticket'] == '113572'], df_test[df_test['Ticket'] == '113572']


# In[23]:


# so, lets take the mode emberking of first class passengers

embarked_mode = df[df['Pclass'] == 1]['Embarked'].mode()[0]

df['Embarked'] = df['Embarked'].fillna(embarked_mode)

embarked_mode


# In[24]:


#
# i will come back to the rest of the missing data as i get to it
# for now, i want to explore the data
#


# In[25]:


# Survived

sns.countplot(x='Survived', data=df)


# In[26]:


# Pclass

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Pclass', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Pclass', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Pclass', y = 'Survived', data=df, ax=ax[1,0])


# In[27]:


# for Pclass, the test data has proportionally a little less third class
# survival rate is better for first class > second class > third class


# In[28]:


# sex

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Sex', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Sex', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Sex', y = 'Survived', data=df, ax=ax[1,0])


# In[29]:


# the proportion of males vs females in the training and test data is biased to females in the test data
# clearly, females had a much higher survival rate


# In[30]:


# exploring Sex and Pclass a bit more

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Pclass', data=df, ax=ax[0,0])
sns.countplot(x='Pclass', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Pclass', y = 'Survived', data=df, ax=ax[1,0])
sns.barplot(x='Pclass', y = 'Survived', hue = 'Sex', data=df, ax=ax[1,1])


# In[31]:


# in first and second class there are slightly more males than females
# in third class, males dominate
# survival rates are biased to females in all classes
# though females are a proportionally much lower survival rate in third class


# In[32]:


# Age - the missing data
# i want to investigate if the missing data itself gives information

df['missing age'] = np.where(df['Age'].isnull(), 1,0)
df_test['missing age'] = np.where(df_test['Age'].isnull(), 1,0)

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='missing age', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='missing age', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='missing age', y = 'Survived', data=df, ax=ax[1,0])


# In[33]:


# similar weightings of missing age in training and test data
# and the survival rate is lower


# In[34]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='missing age', y = 'Survived', data=df, ax=ax[0,0])
sns.barplot(x='missing age', y = 'Survived', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='missing age', y = 'Survived', hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='missing age', y = 'Survived', hue = 'Embarked', data=df, ax=ax[1,1])


# In[35]:


# by Sex and Class, missing Age seems to indecate lower survivability except in class 3


# In[36]:


# Age - more

fig, ax =plt.subplots(figsize=(22,9))
ax = sns.countplot(x='Age', data=df)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
plt.show()


# In[37]:


fig, ax =plt.subplots(figsize=(22,9))
ax = sns.countplot(x='Age', data=df_test)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
plt.show()


# In[38]:


# although it looks like the test data is skewed away from the young, this is not the case
# though it is slightly

df[df['Age'] < 17].count()[0]/len(df), df_test[df_test['Age'] < 17].count()[0]/len(df_test)


# In[39]:


# further down, i establish that the title 'Master' refers to boys up to the age of 15

# so, i will cut the age into years of 5 and see how that looks

df['Age_bins'] = pd.cut(df['Age'],int(df['Age'].max()/5))
df_test['Age_bins'] = pd.cut(df_test['Age'],int(df_test['Age'].max()/5))

fig, ax =plt.subplots(1,2,figsize=(22,9))
sns.countplot(x='Age_bins', data=df, ax=ax[0]).set_title('training data')
sns.countplot(x='Age_bins', data=df_test, ax=ax[1]).set_title('testing data')


# In[40]:


fig, ax =plt.subplots(1,2,figsize=(22,9))
sns.countplot(x='Age_bins', hue = 'Pclass', data=df, ax=ax[0]).set_title('training data')
sns.countplot(x='Age_bins', hue = 'Pclass', data=df_test, ax=ax[1]).set_title('testing data')


# In[41]:


fig, ax =plt.subplots(figsize=(22,9))
sns.barplot(x='Age_bins', y = 'Survived', hue = 'Pclass', data=df)


# In[42]:


# thats interesting, not as many children under the age of 5 and 10 survived as i thought
# i am guessing, but i suspect third class children died with their parenets?


# In[43]:


fig, ax =plt.subplots(figsize=(22,9))
sns.barplot(x='Age_bins', y = 'Survived', hue = 'Sex', data=df)


# In[44]:


# however, being female helped in almost all age brackets
# but being 10 or under helped males

# so it seems age is a useful feature but can be misleading


# In[45]:


# Age
# we have some missing values here
# but i also noticed something odd in the training data (a 14 year old was a Mrs)

# taking a look at the missing values

df[df['Age'].isnull()][0:30]


# In[46]:


null_values = df[df['Age'].isnull()].count()[0]
null_survived = df[(df['Age'].isnull()) & (df['Survived'] == 1)].count()[0]
null_class1 = df[(df['Age'].isnull()) & (df['Pclass'] == 1)].count()[0]
null_class2 = df[(df['Age'].isnull()) & (df['Pclass'] == 2)].count()[0]
null_class3 = df[(df['Age'].isnull()) & (df['Pclass'] == 3)].count()[0]
null_male = df[(df['Age'].isnull()) & (df['Sex'] == 'male')].count()[0]
("")
print(('total missing age ',null_values, '; which survived', null_survived, '; or in class1', null_class1, 
      '; or in class2', null_class2, '; or in class3', null_class3, '; or male', null_male))


# In[47]:


# there seems to be no obvious reason for missing ages
# 52 survived though most died, some in first class though most in third class
# most are male


# In[48]:


# looking at the Age data that is there

df[~df['Age'].isnull()][0:10]


# In[49]:


# i see on line 9 a 'Mrs' who is 14.  Possible, but lets investigate further
# i will need to pull out the titles sooner than i thought

df['Title'] = df['Name'].str.rsplit(',').str[1].str.rsplit('.').str[0].str.strip()
df_test['Title'] = df_test['Name'].str.rsplit(',').str[1].str.rsplit('.').str[0].str.strip()


# In[50]:


# check to make sure i have no created null values

df['Title'].isnull().sum(), df_test['Title'].isnull().sum()


# In[51]:


#df_age_data = df['Title'].value_counts()

data = {'name': df['Title'].value_counts().index,
       'count_training': list(df['Title'].value_counts())}

df_age_data = pd.DataFrame(data)

df_age_data.set_index('name', inplace=True)

df_age_gp1 = df.groupby('Title')['Age'].min()
df_age_gp2 = df.groupby('Title')['Age'].max()
df_age_gp3 = df_test['Title'].value_counts()
df_age_gp4 = df_test.groupby('Title')['Age'].min()
df_age_gp5 = df_test.groupby('Title')['Age'].max()

df_age_data = pd.concat([df_age_data, df_age_gp1,df_age_gp2, df_age_gp3,
                        df_age_gp4, df_age_gp5], axis=1)

df_age_data.columns = ['training count', 'min age', 'max age',
                      'test count', 'min age', 'max age']

df_age_data.sort_values('training count', ascending=False)


# In[52]:


# some items of note:
# Mr goes as young as 11 in the training data, 14 in the test data
# Miss goes as young as 0 in the training and test data
# Mrs hoes as young as 14 in the training data and 16 in the test data
# Masterr goes up to 12 in the training data and 15 in the test data
# many rare titles in the training data are not in the test data

# looking closer at Mr and Mrs at the young ages

df[df['Title'] == 'Mrs']['Age'].sort_values()[0:10]


# In[53]:


df_test[df_test['Title'] == 'Mrs']['Age'].sort_values()[0:10]


# In[54]:


# there are a few young women (under 20) who are classed as Mrs in the training and test data


# In[55]:


df[df['Title'] == 'Mr']['Age'].sort_values()[0:20]


# In[56]:


df_test[df_test['Title'] == 'Mr']['Age'].sort_values()[0:20]


# In[57]:


# there are a few 'Mr' that are under 16


# In[58]:


# question is, which is correct: the title or the age?


# In[59]:


# looking closely at the Mrs under 20 in the training data

df[(df['Title']=='Mrs') & (df['Age']<20)]


# In[60]:


#  all but one survived
# that suggests that they really were young (though being a female also is a big help)


# In[61]:


# looking closely at the Mr under 17 in the training data

df[(df['Title']=='Mr') & (df['Age']<17)]


# In[62]:


# all but one died
# if they were young then i would have expected more survivors
# but they are also mostly third class


# In[63]:


# looking more at the missing age data

# looking at the correlation between 'age' and other data

numerical_data = ['Age', 'Pclass', 'SibSp', 'Parch']

sns.heatmap(df[numerical_data].corr(),annot=True)


# In[64]:


df.shape, df_test.shape


# In[65]:


df.head()


# In[66]:


df_test.tail()


# In[67]:


# Age is negatively correlated with Pclass and SibSp

# but i will also look at other data points, first i need to onehot encode them
# to make sure the onehot encoding assigns the same dummy variables to train and test data i need to combine

train_length = len(df)

df_combined = df
df_combined = df_combined.append(df_test, sort=False).reset_index(drop=True)
one_hot = pd.get_dummies(df_combined[['Sex', 'Embarked']])
df_combined = pd.concat([df_combined, one_hot], axis=1)
df = df_combined[:train_length].reset_index(drop=True)
df_test = df_combined[train_length:].reset_index(drop=True)
df_test.drop('Survived', axis=1, inplace=True)

df.shape, df_test.shape


# In[68]:


numerical_data.extend(['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])


# In[69]:


plt.figure(figsize=(12, 9))
sns.heatmap(df[numerical_data].corr(),annot=True)


# In[70]:


# age is negatively correlated with Pclass - i.e. the first class passengers are older
# age is negatively correlated with SibSp, i.e. brothers and sisters and married people are younger
# other correlations: SibSp and Parch are positive;
# at this point, i am going to look at Title a bit more given it gives me Age information
# reminding myself on the title data:

df_age_data.sort_values('training count', ascending=False)[4:]


# In[71]:


df.head()


# In[72]:


# i will group into: Mr, Miss, Mrs, Master, the rest

rest_list = list((df_age_data.sort_values('training count', ascending=False)[4:]).index)

df['Title_group'] = df['Title']
df['Title_group'] = df['Title_group'].replace(rest_list,'rest')

df_test['Title_group'] = df_test['Title']
df_test['Title_group'] = df_test['Title_group'].replace(rest_list,'rest')

df['Title_group'].value_counts(), df_test['Title_group'].value_counts()


# In[73]:


df_test


# In[74]:


# and now convert the Sex_Label, Embarked_Label and Title_group_Label into numbers

train_length = len(df)

df_combined = df
df_combined = df_combined.append(df_test, sort=False).reset_index(drop=True)
one_hot = pd.get_dummies(df_combined['Title_group'])
df_combined = pd.concat([df_combined, one_hot], axis=1)
df = df_combined[:train_length].reset_index(drop=True)
df_test = df_combined[train_length:].reset_index(drop=True)
df_test.drop('Survived', axis=1, inplace=True)

df.shape, df_test.shape



# df_combined= pd.concat(objs=[df, df_test], axis=0)
# one_hot = pd.get_dummies(df_combined['Title_group'])
# df_combined = pd.concat([df_combined, one_hot], axis=1)
# df = df_combined[:train_length]
# df_test = df_combined[train_length:]
# df_test.drop('Survived', axis=1, inplace=True)

# df.shape, df_test.shape


# In[75]:


df_test.head()


# In[76]:


numerical_data.extend(['Master', 'Miss', 'Mr', 'Mrs'])


# In[77]:


plt.figure(figsize=(12, 9))
sns.heatmap(df[numerical_data].corr(),annot=True, fmt='.2f')


# In[78]:


# age is negatively correlated with Pclass - i.e. the first class passengers are older
# age is negatively correlated with SibSp, i.e. brothers and sisters and married people are younger
# age is negatively correlated with Master and a bit with Miss
# other correlations: SibSp and Parch are positive; SibSp and Master are positve; Parch and Mr are negative


# In[79]:


# so, for age i will use the median value of Pclass, SibSp, Master and Mrs

df['Age'] = df.groupby(['Pclass', 'SibSp', 'Master', 'Mrs'])['Age'].apply(lambda x: x.fillna(x.median()))
df_test['Age'] = df_test.groupby(['Pclass', 'SibSp', 'Master', 'Mrs'])['Age'].apply(lambda x: x.fillna(x.median()))

# this seems to leave the entire Sage family out (one large family)
# so in this case i will give them all an age of 18

df['Age'] = df.groupby(['Pclass', 'SibSp', 'Master', 'Mrs'])['Age'].apply(lambda x: x.fillna(18))
df_test['Age'] = df_test.groupby(['Pclass', 'SibSp', 'Master', 'Mrs'])['Age'].apply(lambda x: x.fillna(18))



# In[80]:


# check to make sure the training and test data has the right shapes

df.shape, df_test.shape


# In[81]:


# will need to complete the age_bins column

df['Age_bins'] = pd.cut(df['Age'],int(df['Age'].max()/5))
df_test['Age_bins'] = pd.cut(df_test['Age'],int(df_test['Age'].max()/5))

fig, ax =plt.subplots(1,2,figsize=(22,9))
sns.countplot(x='Age_bins', data=df, ax=ax[0]).set_title('training data')
sns.countplot(x='Age_bins', data=df_test, ax=ax[1]).set_title('testing data')


# In[82]:


# take another look at Pclass vs Age

fig, ax =plt.subplots(1,2,figsize=(22,9))
sns.countplot(x='Age_bins', hue = 'Pclass', data=df, ax=ax[0]).set_title('training data')
#sns.countplot(x='Age_bins', hue = 'Pclass', data=df_test, ax=ax[1]).set_title('testing data')


# In[83]:


# and survived

fig, ax =plt.subplots(figsize=(22,9))
sns.barplot(x='Age_bins', y = 'Survived', hue = 'Pclass', data=df)


# In[84]:


# and sex

fig, ax =plt.subplots(figsize=(22,9))
sns.barplot(x='Age_bins', y = 'Survived', hue = 'Sex', data=df)


# In[85]:


# interesting that those aged between 5 and 10 did not show good survivorship - lack of first class males
# and indeed males over 10 showed poor survivorship


# In[ ]:





# In[86]:


# looking at SibSp

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='SibSp', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='SibSp', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='SibSp', y = 'Survived', data=df, ax=ax[1,0])


# In[87]:


# noting that 3 or over SibSp hindered survivorship though there are not many counts of those
# having 1 or 2 SibSp was better than none


# In[88]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='SibSp', y = 'Survived', data=df, ax=ax[0,0])
sns.barplot(x='SibSp', y = 'Survived', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='SibSp', y = 'Survived', hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='SibSp', y = 'Survived', hue = 'Embarked', data=df, ax=ax[1,1])


# In[89]:


# male survivorship was improved with 1 SibSp, a little with 2
# Pclass for 1 and 2 meant survivorship better with SibSp of 1,2,3
# for Embarked, Sibsp for 1 was better for all, and 2 for C and Q

# SibSp is siblings and spouses, so i will try and separate the two
# for now, i will assume that anyone under 20 has sibling and is therefore a sibling

df['Sibling or spouse'] = np.where((df['SibSp'] >0) & (df['Age'] <20), 'sibling',
                                   np.where((df['SibSp'] >0) & (df['Age'] >20), 'spouse', 'single'))
df_test['Sibling or spouse'] = np.where((df_test['SibSp'] >0) & (df_test['Age'] <20), 'sibling',
                                   np.where((df_test['SibSp'] >0) & (df_test['Age'] >20), 'spouse', 'single'))


# In[90]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Sibling or spouse', order = ['single', 'sibling', 'spouse'], data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Sibling or spouse', order = ['single', 'sibling', 'spouse'], data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Sibling or spouse', y = 'Survived', order = ['single', 'sibling', 'spouse'], data=df, ax=ax[1,0])


# In[91]:


# many more single people and more spouses vs siblings as passengers in traning and test data
# interestingly, spouses had the highest survival rate


# In[92]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='Sibling or spouse', y = 'Survived', order = ['single', 'sibling', 'spouse'], data=df, ax=ax[0,0])
sns.barplot(x='Sibling or spouse', y = 'Survived', order = ['single', 'sibling', 'spouse'], hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Sibling or spouse', y = 'Survived', order = ['single', 'sibling', 'spouse'], hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='Sibling or spouse', y = 'Survived', order = ['single', 'sibling', 'spouse'], hue = 'Embarked', data=df, ax=ax[1,1])


# In[93]:


# male siblings have a higher survival rate though females lower
# female spouses have as high a survival rate as single females
# in all classes, siblings have a better survival rate
# in third class, survival rate is lower for spouses


# In[94]:


# now i will look at the number of siblings on board per family

df['Sibling size'] = np.where((df['Sibling or spouse'] == 'sibling'), df['SibSp'] + 1, 0)
df_test['Sibling size'] = np.where((df_test['Sibling or spouse'] == 'sibling'), df_test['SibSp'] + 1, 0)


# In[95]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Sibling size', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Sibling size', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Sibling size', y = 'Survived', data=df, ax=ax[1,0])


# In[96]:


# 2 and 3 siblings had a good survival rate, 4 and 5 poor


# In[97]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='Sibling size', y = 'Survived', data=df, ax=ax[0,0])
sns.barplot(x='Sibling size', y = 'Survived', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Sibling size', y = 'Survived', hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='Sibling size', y = 'Survived', hue = 'Embarked', data=df, ax=ax[1,1])


# In[98]:


# males had a better survival rate in 2&3 sibling bucket
# all classes better in 2 and 3


# In[99]:


# looking at Parch

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Parch', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Parch', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Parch', y = 'Survived', data=df, ax=ax[1,0])


# In[100]:


# i will need to split out the Parents from the children as i did with SibSp
# i will assume a Parent is over 20

df['Parent or child'] = np.where((df['Parch'] >0) & (df['Age'] <20), 'child',
                                   np.where((df['Parch'] >0) & (df['Age'] >20), 'parent', 'not parch'))
df_test['Parent or child'] = np.where((df_test['Parch'] >0) & (df_test['Age'] <20), 'child',
                                   np.where((df_test['Parch'] >0) & (df_test['Age'] >20), 'parent', 'not parch'))


# In[101]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Parent or child', order = ['not parch', 'child', 'parent'], data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Parent or child', order = ['not parch', 'child', 'parent'], data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Parent or child', y = 'Survived', order = ['not parch', 'child', 'parent'], data=df, ax=ax[1,0])


# In[102]:


# a child as a better survival rate but not far behind a parent
# the child survival rate is better than for siblings which would indicate that a child with no siblings did better
# i will look into this soon


# In[103]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='Parent or child', y = 'Survived', order = ['not parch', 'child', 'parent'], data=df, ax=ax[0,0])
sns.barplot(x='Parent or child', y = 'Survived', order = ['not parch', 'child', 'parent'], hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Parent or child', y = 'Survived', order = ['not parch', 'child', 'parent'], hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='Parent or child', y = 'Survived', order = ['not parch', 'child', 'parent'], hue = 'Embarked', data=df, ax=ax[1,1])


# In[104]:


# male children had better survival, female children worse, as per siblings
# better survival in all classes, parents worse in third class


# In[105]:


# i will now look for families with one child

df['one child'] = np.where((df['Parch'] >0) & (df['Age'] <20) & (df['SibSp'] == 0), 1,0)
df_test['one child'] = np.where((df_test['Parch'] >0) & (df_test['Age'] <20) & (df_test['SibSp'] == 0), 1,0)

# and add this into the 'Sibling size' column

df['Sibling size'] = df['Sibling size'] + df['one child']
df_test['Sibling size'] = df_test['Sibling size'] + df_test['one child']

# and redo the charts again


# In[106]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Sibling size', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Sibling size', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Sibling size', y = 'Survived', data=df, ax=ax[1,0])


# In[107]:


# single children had the highest survival rate


# In[108]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='Sibling size', y = 'Survived', data=df, ax=ax[0,0])
sns.barplot(x='Sibling size', y = 'Survived', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Sibling size', y = 'Survived', hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='Sibling size', y = 'Survived', hue = 'Embarked', data=df, ax=ax[1,1])


# In[109]:


# single male children had very good survival rates, single female children not much different


# In[110]:


# looking at Ticket
# now, here we have a problem.  there total passengers and crew on titanic was 2224
# we only have 1309 - so we dont have the complete data
# so we need to be careful when we look at the ticket data

# however, i want to look at passengers who travel using the same ticket
# this will cover families and groups

# to get as much information on mulitple passengers on the same ticket, i will also look at the test data
# to get as much info as possible on which tickets are for more than one passenger

train_length = len(df)

df_combined = df
df_combined = df_combined.append(df_test, sort=False).reset_index(drop=True)

df_combined['Ticket count'] = df_combined.groupby('Ticket')['Ticket'].transform('count')

df = df_combined[:train_length].reset_index(drop=True)
df_test = df_combined[train_length:].reset_index(drop=True)
df_test.drop('Survived', axis=1, inplace=True)

df.shape, df_test.shape


# In[111]:


df['Ticket count'].value_counts()


# In[112]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Ticket count', data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Ticket count', data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Ticket count', y = 'Survived', data=df, ax=ax[1,0])


# In[113]:


# tickets were for one passenger in the training and test data
# but survivability looks higer for groups of up to 5 per ticket


# In[114]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='Ticket count', y = 'Survived', data=df, ax=ax[0,0])
sns.barplot(x='Ticket count', y = 'Survived', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Ticket count', y = 'Survived', hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='Ticket count', y = 'Survived', hue = 'Embarked', data=df, ax=ax[1,1])


# In[115]:


# for groups up tp 4, survivability is better for males and females
# and by Pclass and embarked


# In[116]:


# i will create a feature to gather these up

df['Ticket group'] = df['Ticket count']
df['Ticket group'] = df['Ticket group'].replace(1,'single')
df['Ticket group'] = df['Ticket group'].replace([2,3,4],'small group')
df['Ticket group'] = df['Ticket group'].replace([5,6,7,8,9,10,11],'large group')

df_test['Ticket group'] = df_test['Ticket count']
df_test['Ticket group'] = df_test['Ticket group'].replace(1,'single')
df_test['Ticket group'] = df_test['Ticket group'].replace([2,3,4],'small group')
df_test['Ticket group'] = df_test['Ticket group'].replace([5,6,7,8,9,10,11],'large group')


# In[117]:


# now looking for non family groups

df['Ticket non family group'] = np.where((df['Parch'] == 0) & (df['SibSp'] == 0) & (df['Ticket count'] != 1), 'non family group',
                                         np.where((df['Parch'] == 0) & (df['SibSp'] == 0) & (df['Ticket count'] == 1), 'single person ticket','family group'))

df_test['Ticket non family group'] = np.where((df_test['Parch'] == 0) & (df_test['SibSp'] == 0) & (df_test['Ticket count'] != 1), 'non family group',
                                         np.where((df_test['Parch'] == 0) & (df_test['SibSp'] == 0) & (df_test['Ticket count'] == 1), 'single person ticket','family group'))



# In[118]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
order = ['single person ticket', 'non family group','family group' ]
sns.countplot(x='Ticket non family group', data=df, order = order, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Ticket non family group', data=df_test, order = order, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Ticket non family group', y = 'Survived', data=df, order = order, ax=ax[1,0])


# In[119]:


# family groups have the best survival but non family groups are better than single passengers


# In[120]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
order = ['single person ticket', 'non family group','family group' ]
sns.barplot(x='Ticket non family group', y = 'Survived', order = order, data=df, ax=ax[0,0])
sns.barplot(x='Ticket non family group', y = 'Survived', hue = 'Sex', order = order, data=df, ax=ax[0,1])
sns.barplot(x='Ticket non family group', y = 'Survived', hue = 'Pclass', order = order, data=df, ax=ax[1,0])
sns.barplot(x='Ticket non family group', y = 'Survived', hue = 'Embarked', order = order, data=df, ax=ax[1,1])


# In[121]:


# males dont benefit by being in a non-family group but females do
# varies by class


# In[122]:


# Fare has the same problem
# the fare is the fare for the ticket, not the passenger
# given we dont really know how many passengers per ticket, i will skip look at fare


# In[123]:


# cabins - there is too much missing data so i will ignore this feature


# In[124]:


# Embarked i have looked at that closey with the other features to date


# In[125]:


# Title

fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.countplot(x='Title_group', order = ['Master', 'Miss', 'Mr', 'Mrs', 'rest'], data=df, ax=ax[0,0]).set_title('training data')
sns.countplot(x='Title_group', order = ['Master', 'Miss', 'Mr', 'Mrs', 'rest'], data=df_test, ax=ax[0,1]).set_title('test data')
sns.barplot(x='Title_group', y = 'Survived', order = ['Master', 'Miss', 'Mr', 'Mrs', 'rest'], data=df, ax=ax[1,0])


# In[126]:


# similar distribution in training and test data
# Master has the best male survivorship and Mrs is better than miss


# In[127]:


fig, ax =plt.subplots(2,2,figsize=(22,9))
sns.barplot(x='Title_group', y = 'Survived', data=df, ax=ax[0,0])
sns.barplot(x='Title_group', y = 'Survived', hue = 'Sex', data=df, ax=ax[0,1])
sns.barplot(x='Title_group', y = 'Survived', hue = 'Pclass', data=df, ax=ax[1,0])
sns.barplot(x='Title_group', y = 'Survived', hue = 'Embarked', data=df, ax=ax[1,1])


# In[128]:


sns.countplot(x='Title_group', hue = 'Pclass', data=df)


# In[129]:


# the reason for the slightly worse Miss survivorship is that there are more of them in class 3


# In[130]:


# i want to explore how families look if the mother dies
# build a list of ticket numbers with dead mothers:
# females and Parch >0 and Age > 25

dead_mother_tickets = list(df[(df['Sex'] == 'female') & (df['Survived'] == 0) & (df['Parch'] > 0) & (df['Age'] > 25)]['Ticket'])

df['child mother is dead'] = np.where((df['Ticket'].isin(dead_mother_tickets) & (df['Age']<20)),1,0)
df_test['child mother is dead'] = np.where((df_test['Ticket'].isin(dead_mother_tickets) & (df_test['Age']<20)),1,0)


# In[131]:


# Fare - ok, i will take a look at Fare and see if it helps

# firsly, create a new column that adjsuts the fare for the ticket count

df['adj fare'] = df['Fare']/df['Ticket count']
df_test['adj fare'] = df_test['Fare']/df_test['Ticket count']


# In[132]:


# bucket the fares

df['Fare_bins'] = pd.cut(df['adj fare'],int(df['adj fare'].max()/5))
df_test['Fare_bins'] = pd.cut(df_test['adj fare'],int(df_test['adj fare'].max()/5))

fig, ax =plt.subplots(1,2,figsize=(22,9))
sns.countplot(x='Fare_bins', data=df, ax=ax[0]).set_title('training data')
sns.countplot(x='Fare_bins', data=df_test, ax=ax[1]).set_title('testing data')


# In[133]:


# data spread seems similar in trainging and test


# In[134]:


# what is the zero Fare?

df[df['Fare']==0]


# In[135]:


# this looks like crew and servants
# i am going to leave this for now


# In[136]:


# take another look at Pclass vs Fare

fig, ax =plt.subplots(1,2,figsize=(22,9))
sns.countplot(x='Fare_bins', hue = 'Pclass', data=df, ax=ax[0]).set_title('training data')
#sns.countplot(x='Fare_bins', hue = 'Pclass', data=df_test, ax=ax[1]).set_title('testing data')


# In[137]:


# so some overlap of second and third class


# In[138]:


# and sex

fig, ax =plt.subplots(figsize=(22,9))
sns.barplot(x='Fare_bins', y = 'Survived', hue = 'Sex', data=df)


# In[139]:


fig, ax =plt.subplots(figsize=(22,9))
sns.barplot(x='Fare_bins', y = 'Survived', hue = 'Pclass', data=df)


# In[140]:


# might help in the data, so keep it for now


# In[ ]:





# In[ ]:





# In[ ]:





# In[141]:


df.head()


# In[142]:


train_length = len(df)

df_combined = df
df_combined = df_combined.append(df_test, sort=False).reset_index(drop=True)
one_hot = pd.get_dummies(df_combined[['Sibling or spouse', 'Parent or child',
                                     'Ticket group', 'Ticket non family group']])
df_combined = pd.concat([df_combined, one_hot], axis=1)
df = df_combined[:train_length].reset_index(drop=True)
df_test = df_combined[train_length:].reset_index(drop=True)
df_test.drop('Survived', axis=1, inplace=True)

df.shape, df_test.shape


# In[143]:


drop_columns = ['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 
                'Cabin', 'Embarked', 'Title', 'Title_group', 'Sibling or spouse',
               'Parent or child', 'one child', 'Ticket count',
               'Ticket group', 'Ticket non family group', 'adj fare']

df = df.drop(columns=drop_columns)
df_test = df_test.drop(columns=drop_columns)


# In[144]:


features = ['Age_bins', 'Fare_bins']

for feature in features:
    df[feature + str('_Label')] = LabelEncoder().fit_transform(df[feature])
    df_test[feature + str('_Label')] = LabelEncoder().fit_transform(df_test[feature])

df = df.drop(columns=['Age_bins', 'Fare_bins'])
df_test = df_test.drop(columns=['Age_bins', 'Fare_bins'])


# In[145]:


df.shape, df_test.shape


# In[146]:


df.head()


# In[147]:


# drop_columns = []

# df = df.drop(columns=drop_columns)
# df_test = df_test.drop(columns=drop_columns)


# In[ ]:





# In[148]:


#
# Modelling - review a series of models
# use kfolds
#


# In[149]:


# note which trained models are to use the pickled versions


rf_clf_to_load = 0
et_clf_to_load = 0
gb_clf_to_load = 0
bg_clf_to_load = 0
knn_clf_to_load = 0
gnb_clf_to_load = 0
svc_clf_to_load = 0
lr_clf_to_load = 0
mlp_clf_to_load = 0
lda_clf_to_load = 0
cb_clf_to_load = 0


# if not loading a pickled version, determine which trained models are to be pickled
# rf_clf_to_save = 1 - rf_clf_to_load
# et_clf_to_save = 1 - et_clf_to_load
# gb_clf_to_save = 1 - gb_clf_to_load
# bg_clf_to_save = 1 - bg_clf_to_load
# knn_clf_to_save = 1 - knn_clf_to_load
# gnb_clf_to_save = 1 - gnb_clf_to_load
# svc_clf_to_save = 1 - svc_clf_to_load
# lr_clf_to_save = 1 - lr_clf_to_load
# mlp_clf_to_save = 1 - mlp_clf_to_load
# lda_clf_to_save = 1 - lda_clf_to_load
# cb_clf_to_save = 1 - cb_clf_to_load

rf_clf_to_save = 0
et_clf_to_save = 0
gb_clf_to_save = 0
bg_clf_to_save = 0
knn_clf_to_save = 0
gnb_clf_to_save = 0
svc_clf_to_save = 0
lr_clf_to_save = 0
mlp_clf_to_save = 0
lda_clf_to_save = 0
cb_clf_to_save = 0



# In[150]:


random_state=0

ml_clf = MLPClassifier(random_state=random_state)

lr_clf = LogisticRegression(random_state = random_state)

df_clf = DecisionTreeClassifier(random_state=random_state)

ab_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=random_state),
                            random_state=random_state,
                           learning_rate=0.1)

et_clf = ExtraTreesClassifier(random_state=random_state)

lda_clf = LinearDiscriminantAnalysis()

knn_clf = KNeighborsClassifier()

gnb_clf = GaussianNB()

svc_clf = SVC(probability=True,
              random_state=random_state)

bg_clf = BaggingClassifier(random_state=random_state)

gb_clf = GradientBoostingClassifier(random_state=random_state)

cb_clf = CatBoostClassifier(random_state=random_state)

rf_clf = RandomForestClassifier(random_state=0)


# In[151]:


X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived'].astype(int)
X_test = df_test.drop(['PassengerId'], axis=1)

# X = df.drop(['PassengerId', 'Survived'], axis=1)
# y = df['Survived'].astype(int)
# X_test = df_test.drop(['PassengerId'], axis=1)

# X = StandardScaler().fit_transform(df.drop(['PassengerId', 'Survived'], axis=1))
# y = df['Survived'].values
# X_test = StandardScaler().fit_transform(df_test.drop(['PassengerId'], axis=1))


# In[152]:


X.shape, y.shape, X_test.shape


# In[153]:


X.head()


# In[154]:


# looking at the feature importances for a random forest model

# stack = [ml_clf, lr_clf, df_clf, ab_clf, et_clf, lda_clf, knn_clf, gnb_clf, svc_clf,
#         bg_clf, gb_clf, cb_clf, rf_clf]

clf = rf_clf

clf = clf.fit(X, y)

importances = pd.DataFrame({'Features': X.columns, 
                            'Importances': clf.feature_importances_})

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

fig = plt.figure(figsize=(14, 4))
sns.barplot(x='Features', y='Importances', data=importances)
plt.xticks(rotation='vertical')
plt.show()


# In[155]:


# select the key features

edge = 0.02

c = importances.Importances >= edge

columns = importances[c].Features.values

columns


# In[156]:


#columns.append()


# In[157]:


#X = df[columns]
#y = df['Survived'].astype(int)
#X_test = df_test[columns]

X = StandardScaler().fit_transform(df[columns])
y = df['Survived'].values
X_test = StandardScaler().fit_transform(df_test[columns])


# In[158]:


X


# In[159]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[160]:


stack = [ml_clf, lr_clf, df_clf, ab_clf, et_clf, lda_clf, knn_clf, gnb_clf, svc_clf,
        bg_clf, gb_clf, cb_clf, rf_clf]


# In[161]:


clf_accuracy = []
for clf in stack :
    clf_accuracy.append(cross_val_score(clf, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1))


# In[162]:


clf_accuracy_mean = []
clf_accuracy_std = []
for item in clf_accuracy:
    clf_accuracy_mean.append(item.mean())
    clf_accuracy_std.append(item.std())


# In[163]:


stack_names = ['ml_clf', 'lr_clf', 'df_clf', 'ab_clf', 'et_clf', 'lda_clf', 'knn_clf', 'gnb_clf', 'svc_clf', 
               'bg_clf', 'gb_clf', 'cb_clf', 'rf_clf']

df_clf = pd.DataFrame({"clf_accuracy_mean":clf_accuracy_mean,"clf_accuracy_std": clf_accuracy_std,"Classifier":stack_names})


# In[164]:


df_clf


# In[165]:


sns.barplot(x="clf_accuracy_mean", y="Classifier", data=df_clf, orient = "h",**{'xerr':clf_accuracy_std})


# In[ ]:





# In[166]:


#
# fine tuning each model in turn to get better accuracy
#


# In[167]:


# i will use these common inputs for the classifiers and add where required

n_estimators_grid = [10, 50, 100, 300, 600]
max_features_grid = [1, 3, 10, 'auto']
max_samples_grid = [0.1, 0.25, 0.5, 0.75, 1.0]
learning_rate_grid = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5] 
max_depth_grid = [2, 4, 6, 8, 10, None]
min_samples_split_grid = [2, 3, 5, 10, 15]
min_samples_leaf_grid = [1, 3, 10, 15]
criterion_grid = ['gini', 'entropy']
oob_score_grid = [False]
bootstrap_grid = [True]
bool_grid = [True, False]
random_state = [0]


# In[168]:


# # randomforest

# if rf_clf_to_load != 1:
    
#     rf_clf = RandomForestClassifier()


#     ## parameters to trial
#     rf_param_grid = {"max_depth": max_depth_grid,
#                   "max_features": max_features_grid,
#                   "min_samples_split": min_samples_split_grid,
#                   "min_samples_leaf": min_samples_leaf_grid,
#                   "bootstrap": bootstrap_grid,
#                   "n_estimators" : n_estimators_grid,
#                      "oob_score": oob_score_grid,
#                   "criterion": criterion_grid,
#                     "random_state": random_state}


#     rf_clf_grid = GridSearchCV(rf_clf, param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

#     rf_clf_grid.fit(X,y)

#     rf_clf_best = rf_clf_grid.best_estimator_

#     # Best score
#     print(rf_clf_grid.best_score_)

# else:
#     print('using a pickled version')
#     clf_to_load = open('rf_clf_best_Saved','rb')
#     rf_clf_best = pickle.load(clf_to_load)
#     clf_to_load.close()
    


# In[169]:


# # show the parameters of the best rf model
# rf_clf_best


# In[170]:


# # show its accuracy
# cross_val_score(rf_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[171]:


# # save the best model for faster loading next run

# if rf_clf_to_save == 1: # 1 to save, 0 not to save
#     print('saving a pickled version of the stack')
#     clf_best_Saved = open('rf_clf_best_Saved','wb')
#     pickle.dump(rf_clf_best,clf_best_Saved)
#     clf_best_Saved.close()
# else:
#     print('not saving a pickled version of the model')


# In[172]:


# # extratrees

# if et_clf_to_load != 1:
    
#     et_clf = ExtraTreesClassifier()


#     ## parameters to trial
#     et_param_grid = {"max_depth": max_depth_grid,
#                   "max_features": max_features_grid,
#                   "min_samples_split": min_samples_split_grid,
#                   "min_samples_leaf": min_samples_leaf_grid,
#                   "bootstrap": bootstrap_grid,
#                   "n_estimators" :n_estimators_grid,
#                   "criterion": criterion_grid}

#     et_clf_grid = GridSearchCV(et_clf, param_grid = et_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

#     et_clf_grid.fit(X,y)

#     et_clf_best = et_clf_grid.best_estimator_

#     # Best score
#     print(et_clf_grid.best_score_)

# else:
#     print('using a pickled version')
#     clf_to_load = open('et_clf_best_Saved','rb')
#     et_clf_best = pickle.load(clf_to_load)
#     clf_to_load.close()


# In[173]:


# et_clf_best


# In[174]:


# # show its accuracy
# cross_val_score(et_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[175]:


# # save the best model for faster loading next run

# if et_clf_to_save == 1: # 1 to save, 0 not to save
#     print('saving a pickled version of the stack')
#     clf_best_Saved = open('et_clf_best_Saved','wb')
#     pickle.dump(et_clf_best,clf_best_Saved)
#     clf_best_Saved.close()
# else:
#     print('not saving a pickled version of the model')


# In[176]:


# # gradientboosting

# if gb_clf_to_load != 1:
    
#     gb_clf = GradientBoostingClassifier()


#     ## parameters to trial
#     gb_param_grid = {'loss' : ["deviance"],
#                   'n_estimators' : n_estimators_grid,
#                   'learning_rate': learning_rate_grid,
#                   'max_depth': max_depth_grid,
#                   'min_samples_leaf': min_samples_leaf_grid,
#                   'max_features': max_features_grid }

#     gb_clf_grid = GridSearchCV(gb_clf, param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

#     gb_clf_grid.fit(X,y)

#     gb_clf_best = gb_clf_grid.best_estimator_

#     # Best score
#     print(gb_clf_grid.best_score_)

# else:
#     print('using a pickled version')
#     clf_to_load = open('gb_clf_best_Saved','rb')
#     gb_clf_best = pickle.load(clf_to_load)
#     clf_to_load.close()


# In[177]:


# gb_clf_best


# In[178]:


# # show its accuracy
# cross_val_score(gb_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[179]:


# # save the best model for faster loading next run

# if gb_clf_to_save == 1: # 1 to save, 0 not to save
#     print('saving a pickled version of the stack')
#     clf_best_Saved = open('gb_clf_best_Saved','wb')
#     pickle.dump(gb_clf_best,clf_best_Saved)
#     clf_best_Saved.close()
# else:
#     print('not saving a pickled version of the model')


# In[180]:


# bagging

if bg_clf_to_load != 1:

    bg_clf = BaggingClassifier()

    ## parameters to trial
    bg_param_grid = {'n_estimators': n_estimators_grid,
                     'max_samples': max_samples_grid,
                     'random_state': random_state}

    bg_clf_grid = GridSearchCV(bg_clf, param_grid = bg_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    bg_clf_grid.fit(X,y)

    bg_clf_best = bg_clf_grid.best_estimator_

    # Best score
    print((bg_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('bg_clf_best_Saved','rb')
    bg_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[181]:


bg_clf_best


# In[182]:


# show its accuracy
cross_val_score(bg_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[183]:


# save the best model for faster loading next run

if bg_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('bg_clf_best_Saved','wb')
    pickle.dump(bg_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[184]:


# knn

if knn_clf_to_load != 1:

    knn_clf = KNeighborsClassifier()

    ## parameters to trial
    knn_param_grid = {'n_neighbors': [1,2,3,4,5,6,7],
                      'leaf_size': [15,20,25,30],
                      'metric': ['minkowski'],
                      'metric_params': [None],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                     'p': [2]}
    
    knn_clf_grid = GridSearchCV(knn_clf, param_grid = knn_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    knn_clf_grid.fit(X,y)

    knn_clf_best = knn_clf_grid.best_estimator_

    # Best score
    print((knn_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('knn_clf_best_Saved','rb')
    knn_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[185]:


knn_clf_best


# In[186]:


# show its accuracy
cross_val_score(knn_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[187]:


# save the best model for faster loading next run

if knn_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('knn_clf_best_Saved','wb')
    pickle.dump(knn_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[188]:


# gnb

if gnb_clf_to_load != 1:

    gnb_clf = GaussianNB()

    ## parameters to trial
    gnb_param_grid = {}

    gnb_clf_grid = GridSearchCV(gnb_clf, param_grid = gnb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    gnb_clf_grid.fit(X,y)

    gnb_clf_best = gnb_clf_grid.best_estimator_

    # Best score
    print((gnb_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('gnb_clf_best_Saved','rb')
    gnb_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[189]:


gnb_clf_best


# In[190]:


# show its accuracy
cross_val_score(gnb_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[191]:


# save the best model for faster loading next run

if gnb_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('gnb_clf_best_Saved','wb')
    pickle.dump(gnb_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[192]:


# svc

if svc_clf_to_load != 1:

    svc_clf = SVC()

    ## parameters to trial
    svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1, 'auto'],
                  'C': [1, 10, 50, 100,200,300, 1000],
                     'probability': [True]}

    svc_clf_grid = GridSearchCV(svc_clf, param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    svc_clf_grid.fit(X,y)

    svc_clf_best = svc_clf_grid.best_estimator_

    # Best score
    print((svc_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('svc_clf_best_Saved','rb')
    svc_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[193]:


svc_clf_best


# In[194]:


# show its accuracy
cross_val_score(svc_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[195]:


# save the best model for faster loading next run

if svc_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('svc_clf_best_Saved','wb')
    pickle.dump(svc_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[196]:


# LogisticRegression

if lr_clf_to_load != 1:

    lr_clf = LogisticRegression()

    ## parameters to trial
    lr_param_grid = {'fit_intercept': bool_grid ,
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'random_state': random_state}

    lr_clf_grid = GridSearchCV(lr_clf, param_grid = lr_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    lr_clf_grid.fit(X,y)

    lr_clf_best = lr_clf_grid.best_estimator_

    # Best score
    print((lr_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('lr_clf_best_Saved','rb')
    lr_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[197]:


lr_clf_best


# In[198]:


# show its accuracy
cross_val_score(lr_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[199]:


# save the best model for faster loading next run

if lr_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('lr_clf_best_Saved','wb')
    pickle.dump(lr_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[200]:


if mlp_clf_to_load != 1:

    mlp_clf = MLPClassifier()

    ## parameters to trial
    mlp_param_grid = {'hidden_layer_sizes': [25,] ,
                     'max_iter': [100,300,600],
                     'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                     'solver' : ['lbfgs', 'sgd', 'adam'],
                     'random_state': random_state}

    mlp_clf_grid = GridSearchCV(mlp_clf, param_grid = mlp_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    mlp_clf_grid.fit(X,y)

    mlp_clf_best = mlp_clf_grid.best_estimator_

    # Best score
    print((mlp_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('mlp_clf_best_Saved','rb')
    mlp_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[201]:


mlp_clf_best


# In[202]:


# show its accuracy
cross_val_score(mlp_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[203]:


# save the best model for faster loading next run

if mlp_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('mlp_clf_best_Saved','wb')
    pickle.dump(mlp_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[204]:


if lda_clf_to_load != 1:

    lda_clf = LinearDiscriminantAnalysis()

    ## parameters to trial
    lda_param_grid = {'solver': ['svd', 'lsqr', 'eigen'],}

    lda_clf_grid = GridSearchCV(lda_clf, param_grid = lda_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    lda_clf_grid.fit(X,y)

    lda_clf_best = lda_clf_grid.best_estimator_

    # Best score
    print((lda_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('lda_clf_best_Saved','rb')
    lda_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[205]:


lda_clf_best


# In[206]:


# show its accuracy
cross_val_score(lda_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[207]:


# save the best model for faster loading next run

if lda_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('lda_clf_best_Saved','wb')
    pickle.dump(lda_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[208]:


if cb_clf_to_load != 1:

    cb_clf = CatBoostClassifier()

    ## parameters to trial
    cb_param_grid = {'iterations': [2, 5, 10, 20, 30, 40, 50],
                     'learning_rate': [0.2, 0.5, 1],
                     'depth': [2, 5, 10, 15, 20],
                     'loss_function': ['Logloss'],
                     'random_state': random_state}
    
    cb_clf_grid = GridSearchCV(cb_clf, param_grid = cb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

    cb_clf_grid.fit(X,y)

    cb_clf_best = cb_clf_grid.best_estimator_

    # Best score
    print((cb_clf_grid.best_score_))
    
else:
    print('using a pickled version')
    clf_to_load = open('cb_clf_best_Saved','rb')
    cb_clf_best = pickle.load(clf_to_load)
    clf_to_load.close()


# In[209]:


cb_clf_best


# In[210]:


# show its accuracy
cross_val_score(cb_clf_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1).mean()


# In[211]:


# save the best model for faster loading next run

if cb_clf_to_save == 1: # 1 to save, 0 not to save
    print('saving a pickled version of the stack')
    clf_best_Saved = open('cb_clf_best_Saved','wb')
    pickle.dump(cb_clf_best,clf_best_Saved)
    clf_best_Saved.close()
else:
    print('not saving a pickled version of the model')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[212]:


# stack_best_names = ['ml_clf_best', 'lr_clf_best', 'df_clf_best', 'ab_clf_best', 'et_clf_best',
#                     'lda_clf_best', 'knn_clf_best', 'gnb_clf_best', 'svc_clf_best',
#                     'bg_clf_best', 'gb_clf_best', 'cb_clf_best', 'rf_clf_best']

stack_best_names = ['ml_clf_best', 'lr_clf_best', 'et_clf',
                    'lda_clf_best', 'knn_clf_best', 'gnb_clf', 'svc_clf_best',
                    'bg_clf_best', 'gb_clf_best', 'cb_clf_best', 'rf_clf', 'ab_clf']

# stack_best = [mlp_clf_best, lr_clf_best, df_clf_best, ab_clf_best, et_clf_best, lda_clf_best, 
#               knn_clf_best, gnb_clf_best, svc_clf_best, bg_clf_best, gb_clf_best, cb_clf_best, 
#               rf_clf_best]

stack_best = [mlp_clf_best, lr_clf_best, et_clf, lda_clf_best, 
              knn_clf_best, gnb_clf_best, svc_clf_best, bg_clf_best, gb_clf, cb_clf_best, 
              rf_clf, ab_clf]

clf_best_accuracy = []
for clf in stack_best:
    clf_best_accuracy.append(cross_val_score(clf, X, y, scoring = "accuracy", cv = kfold, n_jobs=-1))
    
clf_best_accuracy_mean = []
clf_best_accuracy_std = []
for item in clf_best_accuracy:
    clf_best_accuracy_mean.append(item.mean())
    clf_best_accuracy_std.append(item.std())

df_clf_best = pd.DataFrame({"clf_best_accuracy_mean":clf_best_accuracy_mean,"cl_bestf_accuracy_std": clf_best_accuracy_std,
                            "Classifier":stack_best_names})


# In[ ]:





# In[213]:


df_clf_best


# In[214]:


votingC = VotingClassifier(estimators=[('lr',lr_clf), ('lda',lda_clf),('gb', gb_clf) ], voting='hard', n_jobs=-1)

votingC = votingC.fit(X, y)


# In[215]:


test_predictions = votingC.predict(X_test).astype(int)
test_predictions


# In[216]:


output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': test_predictions})
output.to_csv('my_submission_31.csv', index=False)


# In[ ]:





# In[ ]:




