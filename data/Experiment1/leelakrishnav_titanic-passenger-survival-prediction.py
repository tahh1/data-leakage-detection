#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Load Data

# In[2]:


#train set
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

#test set
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


#train_df
print((train_df.shape))


# In[4]:


train_df.head()


# In[5]:


#test_df
print((test_df.shape))

test_df.head()


# We have 418 rows with 11 features in Test dataset
# 
# Now lets clean the data from both train and test dataset,
# 
# For that we will concatenate the both train_df and test_df to maintain homogeneity in this process for both the datasets.

# In[6]:


train_df.describe()


# In[ ]:





# - Average Survival rate is 38%
# - Minimum age of pasenger around 5 months(.042) and maximum age 80, average age at 30 years.
# - 75% passengers travelled alone without any siblings or parents.
# 

# In[ ]:





# ## Distribution

# In[7]:


train_df.describe(include=['O'])


# In[ ]:





# #### What is the distribution of categorical features?
# 
# - Names are unique across the dataset (count=unique=891)
# - Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
# - Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
# - Embarked takes three possible values. S port used by most passengers (top=S)
# - Ticket feature has high ratio (22%) of duplicate values (unique=681).

# In[8]:


train_df.columns


# ### Variable analysis

# In[9]:


train_df.info()


# Categorical Variables:  [Sex, Ticket, Cabin, Embarked, SibSp, Parch,Pclass,]
# 
# Numeric variables : [Survived,  Age, Fare]
#     
# #### Variable Description

# 1. PassengerId: unique id number to each passenger 
# 2. Survived: passenger survive(1) or died(0) 3. Pclass: passenger class 
# 4. Name: name 
# 5. Sex: gender of passenger 
# 6. Age: age of passenger 
# 7. SibSp: number of siblings/spouses 
# 8. Parch: number of parents/children 
# 9. Ticket: ticket number 
# 10. Fare: amount of money spent on ticket 
# 11. Cabin: cabin category 
# 12. Embarked: port where passenger embarked(C = Cherbourg, Q = Queenstown, S = Southampton)

# In[10]:


train_df['Survived'].value_counts()


# We have 342 survivers and 549 non-survivers in our train dataset
# 
# 
# ### Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# Correlating.
# 
# We want to know how well does each feature correlate with Survival. We want to do this early in our project and match these quick correlations with modelled correlations later in the project.
# 
# #### Completing.
# 
#     1. We may want to complete Age feature as it is definitely correlated to survival.
#     2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# #### Correcting.
# 
#     1.Ticket feature may be dropped from our analysis as it contains high ratio of duplicates (22%) and there may not be a correlation between Ticket and survival.
#     2. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
#     3. PassengerId may be dropped from training dataset as it does not contribute to survival.
#     4. Name feature is relatively non-standard, may not contribute directly to survival, so maybe dropped.
# 
# #### Creating.
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.
# 3. We may want to create new feature for Age bands. This turns a continous numerical feature into an ordinal categorical feature.
# 4. We may also want to create a Fare range feature if it helps our analysis.
# 
# #### Classifying.
# 
# We may also add to our assumptions based on the problem description noted earlier.
# 
# 1. Women (Sex=female) were more likely to have survived.
# 2. Children (Age<?) were more likely to have survived.
# 
# 3. The upper-class passengers (Pclass=1) were more likely to have survived.

# Now lets clean the data from both train and test dataset,
# 
# For that we will concatenate the both train_df and test_df to maintain homogeneity in this process for both the datasets.

# In[11]:


df_train_test = pd.concat([train_df, test_df], axis=0, join='outer', ignore_index=False, keys=None,
          levels=None, names=None, verify_integrity=False, copy=True) 
print((df_train_test.shape))

df_train_test.head(10)


# In[12]:


df_train_test.isnull().sum()


# ## Observations
# 
# - Age column as 263 Null values.
# - There are 1014 Nan values in Cabin.
# - Embarked, Fare has 2, 1 respectively.
# - Servived also has shown as 418, this is because this our target feature to be predictedfor test dataset, our test set dont have Servived column(Observed above.). So we wont be dealing with these missing values.
# 
# - This means we have to clean our data as our data contains missing values for Cabin and Age.
# 
# ## Data Cleaning
# 
# Lets identify and drop the columns of such (The onces with no. of. NaN values more than 30%) kind. Because these might effect the aggregations and analysis in our study with these NaN values.
# 
# 
# Drp Cabin colum since 70% values are Nan.

# In[13]:


df_train_test.drop('Cabin',axis=1,inplace=True)


# ### apply the center of mean and median
# 
# #### Age

# In[14]:


#imputing NaNs

df_train_test['Age'] = df_train_test['Age'].fillna(df_train_test['Age'].median())


# #### Embarked

# In[15]:


df_train_test[df_train_test['Embarked'].isnull()]


# In[16]:


print((df_train_test.Embarked.value_counts(dropna=False)))


# the nearest Fare is C.

# In[17]:


# Update Missing Embarked values with C since C has more values.
df_train_test.Embarked.fillna(df_train_test.Embarked.mode()[0],inplace=True)


# In[18]:


print((df_train_test.Embarked.value_counts(dropna=False)))


# In[19]:


df_train_test.isnull().sum()


# #### Fare

# In[20]:


df_train_test[df_train_test['Fare'].isnull()]


# In[21]:


df_train_test.Fare.fillna(df_train_test.Fare.median(),inplace=True)


# ### Spliting data back into train_df and test_df
# 
#   As we know, we can do that in different ways, lets use those NaN values under Survived to this.

# In[22]:


df_train_test[df_train_test.Survived.isnull() == True].shape


# Here we can see the test set after dealing with NaNs. Lets name it back as df_test.

# In[23]:


df_test = df_train_test[df_train_test.Survived.isnull() == True]

df_test = df_test.drop('Survived', axis = 1)

df_test.columns


# In[24]:


df_test.shape


# In[25]:


df_train = df_train_test.dropna(axis = 0)


# In[26]:


df_train.shape


# Here we go, now we have our train set as well.

# ## Feature Analysis
# 
# we can quickly analyze our feature correlations by pivoting features against each other. We can only do so at this stage for features which do not have any empty values. It also makes sense doing so only for features which are categorical (Sex), ordinal (Pclass) or discrete (SibSp, Parch) type.
# 
# - Pclass We observe significant correlation (>0.5) among Pclass=1 and Survived . We decide to include this feature in our model.
# - Sex We confirm the observation during problem definition that Sex=female had very high survival rate at 74% .
# - SibSp and Parch These features have zero correlation for certain values. It may be best to derive a feature or a set of features from these individual features .

# In[27]:


## PClass vs Survived
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[28]:


##Sex vs Survived
df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[29]:


## Sibsp vs Survived
df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[30]:


## Parch vs Survived
df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[31]:


g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ### Observations.
# 
# - Infants (Age <=4) had high survival rate.
# - Oldest passengers (Age = 80) survived.
# - Large number of 15-25 year olds did not survive.
# - Most passengers are in 15-35 age range.
# 
# 
# ### Decisions.
# 
# This simple analysis confirms our assumptions as decisions for subsequent workflow stages.
# 
# - We should consider Age  in our model training.
# - Complete the Age feature for null values.
# - We should band age groups.

# In[32]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ### Correlating categorical features
# 
# Now we can correlate categorical features with our solution goal.
# 
# #### Observations.
# 
# - Female passengers had much better survival rate than males. 
# - Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
# - Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. 
# - Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. 
# 
# #### Decisions.
# 
# - Add Sex feature to model training.
# - Complete and add Embarked feature to model training.

# In[33]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(df_train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[34]:


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# ### Correcting by dropping features

# In[35]:


# since 'sex' is a binary column, each value can be represented as such
# convert the 'sex' col into a binary indicator column

sex_train = pd.get_dummies(df_train['Sex'], drop_first = True) 
# drop_first = True to take away redundant info
# The first column was a perfect predicotr of the second column

# The same could be done to the 'embark' col
embark_train = pd.get_dummies(df_train['Embarked'], drop_first = True)
# Removing one column could remove the 'perfect predictor' aspect


# In[36]:


# Combine the indicator columns with the original dataset and then remove the original columns that were adjusted
df_train_adj = pd.concat([df_train, sex_train, embark_train], axis = 1)
df_train_adj.head()


# In[37]:


df_train_adj.drop('Sex', axis = 1, inplace = True)
df_train_adj.drop('Embarked', axis = 1, inplace = True)
df_train_adj.drop('Ticket', axis = 1, inplace = True)


df_train_adj.head()


# In[38]:


corr_matrix=df_train_adj.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# In[39]:


sex_test = pd.get_dummies(df_test['Sex'], drop_first = True) 
embark_test = pd.get_dummies(df_test['Embarked'], drop_first = True)


# In[40]:


df_test_adj = pd.concat([df_test, sex_test, embark_test], axis = 1)
df_test_adj.head()


# In[41]:


df_test_adj.shape


# In[42]:


df_test_adj.drop('Sex', axis = 1, inplace = True)
df_test_adj.drop('Embarked', axis = 1, inplace = True)
df_test_adj.drop('Ticket', axis = 1, inplace = True)
df_test_adj.head()


# In[43]:


df_train_adj.hist(figsize = (30, 35), bins = 50, xlabelsize = 8, ylabelsize = 8, color='orange');


# We can replace many titles with a more common name or classify them as Rare.

# In[44]:


combine = [df_train_adj, df_test_adj]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train_adj['Title'], df_train_adj['male'])


# In[45]:


combine = [df_train_adj, df_test_adj]

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train_adj[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[46]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

df_train_adj.head()


# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# In[47]:


df_train_adj = df_train_adj.drop(['Name','PassengerId'], axis=1)
df_test_adj = df_test_adj.drop(['Name'], axis=1)
combine = [df_train_adj, df_test_adj]
df_train_adj.shape, df_test_adj.shape


# In[48]:


df_train_adj.head()


# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[49]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(df_train_adj, row='Pclass', col='male', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# Let us start by preparing an empty array to contain guessed Age values based on Pclass x Gender combinations.

# In[50]:


guess_ages = np.zeros((2,3))
guess_ages


# Now we iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[51]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['male'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.male == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

df_train_adj.head()


# Let us create Age bands and determine correlations with Survived.

# In[52]:


df_train_adj['AgeBand'] = pd.cut(df_train_adj['Age'], 5)
df_train_adj[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[53]:


# Mapping Fare
combine = [df_train_adj,df_test_adj]
    
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
df_train_adj.head()


# In[54]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
df_train_adj.head()


# Add future Family_size by combining Parch + SibSP

# In[55]:


df_train_adj["Family_size"] = df_train_adj["SibSp"] + df_train_adj["Parch"]+1
df_train_adj["IsAlone"] = 1
df_train_adj['IsAlone'].loc[df_train_adj['Family_size'] > 1] = 0


# In[56]:


df_test_adj["Family_size"] = df_test_adj["SibSp"] + df_test_adj["Parch"]+1
df_test_adj["IsAlone"] = 1
df_test_adj['IsAlone'].loc[df_test_adj['Family_size'] > 1] = 0


# In[57]:


df_train_adj = df_train_adj.drop(['SibSp', 'Parch'], axis=1)
df_test_adj = df_test_adj.drop(['SibSp', 'Parch'], axis=1)


# In[58]:


df_train_adj = df_train_adj.drop(['Family_size'], axis=1)
df_test_adj = df_test_adj.drop(['Family_size'], axis=1)


# In[59]:


df_test_adj.columns


# In[60]:


df_train_adj = df_train_adj[['Survived','Pclass','male','Age', 'Fare',   'Q', 'S','IsAlone', 'Title']]
df_test_adj = df_test_adj[['PassengerId','Pclass','male','Age', 'Fare',   'Q', 'S','IsAlone', 'Title']]

train = df_train_adj.values
test = df_test_adj.values


# ## Choosing the Best Model

# In[61]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(X_train, y_train)
		train_predictions = clf.predict(X_test)
		acc = accuracy_score(y_test, train_predictions)
		if name in acc_dict:
			acc_dict[name] += acc
		else:
			acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 10.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# In[62]:


from sklearn.model_selection import train_test_split

predictors = df_train_adj.drop(['Survived'], axis=1)
target = df_train_adj["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# In[63]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[64]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[65]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[66]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[67]:


# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[68]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[69]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[70]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[71]:


# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[72]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[73]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)


# ## Creating Submission File

# In[74]:


#set ids as PassengerId and predict survival 
ids = df_test_adj['PassengerId']
predictions = gbk.predict(df_test_adj.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)


# In[ ]:




