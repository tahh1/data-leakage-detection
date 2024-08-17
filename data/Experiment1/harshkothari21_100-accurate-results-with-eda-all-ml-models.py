#!/usr/bin/env python
# coding: utf-8

# <h1 style="color:green"><center>Don't forget to upvote if you like it! It's free! :)

# **For absoulte beginners, do check the notebook**
# 
# # [Beginners Notebook with EDA](https://www.kaggle.com/harshkothari21/beginners-notebook-90-accuracy-with-eda)

# **- Want to know how to analize dataset using pandas, matplotlib and seaborn?**
# 
# **- Want to know how to solve classification problems!**
# 
# **- Want to know how to train the best model?**
# 
# **- Want to know how to use scaling?**
# 
# **- Want to know how to improve your model accuracy?**
# 
# ## **This notebook will answer all of your questions!**

# # Table of content 
# 
# - EDA
# - Handle Missing Values
# - Feature Engineering
# - linear Regression
# - Logistic Regression
# - Scalling
# - KNN Classifier
# - Support Vector Machine(SVM)
# - Kernelize SVM
# - Decision Tree
# - Random Forest

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import warnings
from statistics import mode
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[2]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# In[3]:


target = train.Survived


# <h1 style='color:red'>EDA (Exploratory Data Analysis)

# In[4]:


train.head()


# Variable Name | Description
# --------------|-------------
# Survived      | Survived (1) or died (0)
# Pclass        | Passenger's class
# Name          | Passenger's name
# Sex           | Passenger's sex
# Age           | Passenger's age
# SibSp         | Number of siblings/spouses aboard
# Parch         | Number of parents/children aboard
# Ticket        | Ticket number
# Fare          | Fare
# Cabin         | Cabin
# Embarked      | Port of embarkation

# In[5]:


print(f'Unique Values in Pclass :{train.Pclass.unique()}')


# In[6]:


print(f'Unique Values in SibSp :{train.SibSp.unique()}')


# Hehe!, null values spotted!

# In[7]:


print(f'Unique Values in Embarked :{train.Embarked.unique()}')


# **Let's look at target feature first**

# In[8]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train.Survived)
plt.title('Number of passenger Survived');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Sex", data=train)
plt.title('Number of passenger Survived');


# **So the plot says we have more number of non-survived people and females are more likely to survived than male!. so, 'Sex' looks like a very strong explanatory variable, and it can be good choice for our model!**

# **Let's first vizualize null values on our training set on graph**

# In[9]:


plt.style.use('seaborn')
plt.figure(figsize=(10,5))
sns.heatmap(train.isnull(), yticklabels = False, cmap='plasma')
plt.title('Null Values in Training Set');


# **We will be dealling with null values later on.**

# **Let's analysize Pclass**

# In[10]:


plt.figure(figsize=(15,5))
plt.style.use('fivethirtyeight')

plt.subplot(1,2,1)
sns.countplot(train['Pclass'])
plt.title('Count Plot for PClass');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Pclass", data=train)
plt.title('Number of passenger Survived');


# looking at some satistical data!

# In[11]:


pclass1 = train[train.Pclass == 1]['Survived'].value_counts(normalize=True).values[0]*100
pclass2 = train[train.Pclass == 2]['Survived'].value_counts(normalize=True).values[1]*100
pclass3 = train[train.Pclass == 3]['Survived'].value_counts(normalize=True).values[1]*100

print("Lets look at some satistical data!\n")
print(("Pclaas-1: {:.1f}% People Survived".format(pclass1)))
print(("Pclaas-2: {:.1f}% People Survived".format(pclass2)))
print(("Pclaas-3: {:.1f}% People Survived".format(pclass3)))


# **Wow!, Pclass is also a good feature to train our model.**

# **It's Time to look at the Age column!**

# In[12]:


train['Age'].plot(kind='hist')


# **Most Important thing when plotting histograms : Arrange Number of Bins**

# In[13]:


train['Age'].hist(bins=40)
plt.title('Age Distribution');


# **Age column has non-uniform data and many outliers**
# 
# **Outlier** : An outlier is an observation that lies an abnormal distance from other values in a random sample from a population.

# In[14]:


# set plot size
plt.figure(figsize=(15, 3))

# plot a univariate distribution of Age observations 
sns.distplot(train[(train["Age"] > 0)].Age, kde_kws={"lw": 3}, bins = 50)

# set titles and labels
plt.title('Distrubution of passengers age',fontsize= 14)
plt.xlabel('Age')
plt.ylabel('Frequency')
# clean layout
plt.tight_layout()


# **Age by surviving status**
# 
# Did age had a big influence on chances to survive?
# To visualize two age distributions, grouped by surviving status I am using boxlot and stripplot showed together:

# In[15]:


plt.figure(figsize=(15, 3))

# Draw a box plot to show Age distributions with respect to survival status.
sns.boxplot(y = 'Survived', x = 'Age', data = train,
     palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')

# Add a scatterplot for each category.
sns.stripplot(y = 'Survived', x = 'Age', data = train,
     linewidth = 0.6, palette=["#3f3e6fd1", "#85c6a9"], orient = 'h')

plt.yticks( np.arange(2), ['drowned', 'survived'])
plt.title('Age distribution grouped by surviving status (train data)',fontsize= 14)
plt.ylabel('Passenger status after the tragedy')
plt.tight_layout()


# **Let's look at Number of siblings/spouses aboard**

# In[16]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['SibSp'])
plt.title('Number of siblings/spouses aboard');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="SibSp", data=train)
plt.legend(loc='right')
plt.title('Number of passenger Survived');


# **Looks like single person Non-survived count is almost double than survived, while others have 50-50 % ratio**

# **Now Looking at Port of embarkation**

# In[17]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(train['Embarked'])
plt.title('Number of Port of embarkation');

plt.subplot(1,2,2)
sns.countplot(x="Survived", hue="Embarked", data=train)
plt.legend(loc='right')
plt.title('Number of passenger Survived');


# **Can't say much!**

# **Look in to relationships among dataset**

# In[18]:


sns.heatmap(train.corr(), annot=True)
plt.title('Corelation Matrix');


# **Configure the heatmap**

# In[19]:


corr = train.corr()
sns.heatmap(corr[((corr >= 0.3) | (corr <= -0.3)) & (corr != 1)], annot=True, linewidths=.5, fmt= '.2f')
plt.title('Configured Corelation Matrix');


# **Fare vs Embarked**

# In[20]:


sns.catplot(x="Embarked", y="Fare", kind="violin", inner=None,
            data=train, height = 6, order = ['C', 'Q', 'S'])
plt.title('Distribution of Fare by Embarked')
plt.tight_layout()


# - The wider fare distribution among passengers who embarked in Cherbourg. It makes scence - many first-class passengers boarded the ship here, but the share of third-class passengers is quite significant.
# - The smallest variation in the price of passengers who boarded in q. Also, the average price of these passengers is the smallest, I think this is due to the fact that the path was supposed to be the shortest + almost all third-class passengers.

# **Fare vs Pclass**

# In[21]:


sns.catplot(x="Pclass", y="Fare", kind="swarm", data=train, height = 6)

plt.tight_layout()


# We can observe that the distribution of prices for the second and third class is very similar. The distribution of first-class prices is very different, has a larger spread, and on average prices are higher.
# 
# Let's add colours to our points to indicate surviving status of passenger (there will be only data from training part of the dataset):

# In[22]:


sns.catplot(x="Pclass", y="Fare",  hue = "Survived", kind="swarm", data=train, 
                                    palette=["#3f3e6fd1", "#85c6a9"], height = 6)
plt.tight_layout()


# Let's look at some maximum and minimum values of features!

# In[23]:


train['Fare'].nlargest(10).plot(kind='bar', title = '10 largest Fare', color = ['#C62D42', '#FE6F5E']);
plt.xlabel('Index')
plt.ylabel('Fare');


# In[24]:


train['Age'].nlargest(10).plot(kind='bar', color = ['#5946B2','#9C51B6']);
plt.title('10 largest Ages')
plt.xlabel('Index')
plt.ylabel('Ages');


# In[25]:


train['Age'].nsmallest(10).plot(kind='bar', color = ['#A83731','#AF6E4D'])
plt.title('10 smallest Ages')
plt.xlabel('Index')
plt.ylabel('Ages');


# <h1 style='color:red'>Handle Missing Values

# Some statistical values of null values in dataset.

# In[26]:


train.isnull().sum()


# In[27]:


test.isnull().sum()


# one of the effectitve way to fill the null values is by finding correlation

# In[28]:


sns.heatmap(train.corr(), annot=True)


# > **Pclass and age, as they had max relation in the entire set we are going to replace missing age values with median age calculated per class**

# In[29]:


train.loc[train.Age.isnull(), 'Age'] = train.groupby("Pclass").Age.transform('median')


#Same thing for test set
test.loc[test.Age.isnull(), 'Age'] = test.groupby("Pclass").Age.transform('median')


# In[30]:


train.Embarked.value_counts()


# > **As maximum values in train set is S let's replace it with the null values**

# In[31]:


train['Embarked'] = train['Embarked'].fillna(mode(train['Embarked']))

#Applying the same technique for test set
test['Embarked'] = test['Embarked'].fillna(mode(test['Embarked']))


# > Also, corr(Fare, Pclass) is the highest correlation in absolute numbers for 'Fare', so we'll use Pclass again to impute the missing values!

# In[32]:


train['Fare']  = train.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))
test['Fare']  = test.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))


# In[33]:


train.Cabin.value_counts()


# > So many different values let's place missing values with U as "Unknown"

# In[34]:


train['Cabin'] = train['Cabin'].fillna('U')
test['Cabin'] = test['Cabin'].fillna('U')


# # Feature Engineering

# In[35]:


train.Sex.unique()


# > Sex is categorical data so we can replace male to 0 and femail to 1

# In[36]:


train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1

test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1


# In[37]:


train.Embarked.unique()


# > Let's encode with OneHotEncoder technique

# In[38]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
train = train.join(temp)
train.drop(columns='Embarked', inplace=True)

temp = pd.DataFrame(encoder.transform(test[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])
test = test.join(temp)
test.drop(columns='Embarked', inplace=True)


# In[39]:


train.columns


# In[40]:


train.Cabin.tolist()[0:20]


# > We can get the alphabets by running regular expression

# In[41]:


train['Cabin'] = train['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
test['Cabin'] = test['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())


# In[42]:


train.Cabin.unique()


# In[43]:


cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}
train['Cabin'] = train['Cabin'].map(cabin_category)
test['Cabin'] = test['Cabin'].map(cabin_category)


# ### What is in the name?
# Each passenger Name value contains the title of the passenger which we can extract and discover.
# To create new variable "Title":
# 
# - I am using method 'split' by comma to divide Name in two parts and save the second part
# - I am splitting saved part by dot and save first part of the result
# - To remove spaces around the title I am using 'split' method
# - To visualize, how many passengers hold each title, I chose countplot.

# In[44]:


train.Name


# In[45]:


train['Name'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
test['Name'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand = False)


# In[46]:


train['Name'].unique().tolist()


# **Wohh that's lot's of title. So, let's bundle them**
# 

# In[47]:


train.rename(columns={'Name' : 'Title'}, inplace=True)
train['Title'] = train['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
                                      
test.rename(columns={'Name' : 'Title'}, inplace=True)
test['Title'] = test['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')


# In[48]:


train['Title'].value_counts(normalize = True) * 100


# Better! let's convert to numeric

# In[49]:


encoder = OneHotEncoder()
temp = pd.DataFrame(encoder.fit_transform(train[['Title']]).toarray())
train = train.join(temp)
train.drop(columns='Title', inplace=True)

temp = pd.DataFrame(encoder.transform(test[['Title']]).toarray())
test = test.join(temp)
test.drop(columns='Title', inplace=True)


# Hmmm... but we know from part 2 that Sibsp is the number of siblings / spouses aboard the Titanic, and Parch is the number of parents / children aboard the Titanic... So, what is another straightforward feature to engineer?
# Yes, it is the size of each family aboard!
# 

# In[50]:


train['familySize'] = train['SibSp'] + train['Parch'] + 1
test['familySize'] = test['SibSp'] + test['Parch'] + 1


# In[51]:


fig = plt.figure(figsize = (15,4))

ax1 = fig.add_subplot(121)
ax = sns.countplot(train['familySize'], ax = ax1)

# calculate passengers for each category
labels = (train['familySize'].value_counts())
# add result numbers on barchart
for i, v in enumerate(labels):
    ax.text(i, v+6, str(v), horizontalalignment = 'center', size = 10, color = 'black')
    
plt.title('Passengers distribution by family size')
plt.ylabel('Number of passengers')

ax2 = fig.add_subplot(122)
d = train.groupby('familySize')['Survived'].value_counts(normalize = True).unstack()
d.plot(kind='bar', color=["#3f3e6fd1", "#85c6a9"], stacked='True', ax = ax2)
plt.title('Proportion of survived/drowned passengers by family size (train data)')
plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))
plt.xticks(rotation = False)

plt.tight_layout()


# In[52]:


# Drop redundant features
train = train.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)
test = test.drop(['SibSp', 'Parch', 'Ticket'], axis = 1)


# In[53]:


train.head()


# ## PCA(Principle component analysis)
# 
# let’s visualize our final dataset by implementing PCA and plot the graph

# In[54]:


columns = train.columns[2:]
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(train.drop(columns=["PassengerId","Survived"]))

new_df = pd.DataFrame(X_train, columns=columns)


# In[55]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
df_pca = pca.fit_transform(new_df)


# In[56]:


plt.figure(figsize =(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c = target, cmap ='plasma')
# labeling x and y axes
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component');


# Ourdataset contain some outliers and randomness but still let's use this to train the model.

# Dateset is completely ready now!
# 

# In[57]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'], test_size = 0.2, random_state=2)


# # Linear Regression
# 
# Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression.

# In[58]:


from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train, y_train)

#R-Squared Score
print(("R-Squared for Train set: {:.3f}".format(linreg.score(X_train, y_train))))
print(("R-Squared for test set: {:.3f}" .format(linreg.score(X_test, y_test))))


# it's clear from the score that linear regression doesn't makes sence

# # Logistic Regression
# 
# As our target variable is discrete value(i.e 0 and 1) logistic regression is more likely to fit well the model

# In[59]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=10000, C=50)
logreg.fit(X_train, y_train)

#R-Squared Score
print(("R-Squared for Train set: {:.3f}".format(logreg.score(X_train, y_train))))
print(("R-Squared for test set: {:.3f}" .format(logreg.score(X_test, y_test))))


# > Haha, much better!

# > **Additionally, you can view the y-intercept and coefficients**

# In[60]:


print((logreg.intercept_))
print((logreg.coef_))


# # MinMaxScaler
# 

# 
# 
# # Magic Weapon#1: **Let's Scale our data and re-train the model**

# In[61]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)


# In[62]:


logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train_scaled, y_train)

#R-Squared Score
print(("R-Squared for Train set: {:.3f}".format(logreg.score(X_train_scaled, y_train))))
print(("R-Squared for test set: {:.3f}" .format(logreg.score(X_test_scaled, y_test))))


# **Let's try some other Techniques**

# # KNN Classifier
# 
# K Nearest Neighbor(KNN) is a very simple, easy to understand, versatile and one of the topmost machine learning algorithms.KNN algorithm used for both classification and regression problems.

# In[63]:


from sklearn.neighbors import KNeighborsClassifier

knnclf = KNeighborsClassifier(n_neighbors=7)

# Train the model using the training sets
knnclf.fit(X_train, y_train)
y_pred = knnclf.predict(X_test)


# In[64]:


from sklearn.metrics import accuracy_score

# Model Accuracy, how often is the classifier correct?
print(("Accuracy:",accuracy_score(y_test, y_pred)))


# ### Let's try on scaled data

# In[65]:


knnclf = KNeighborsClassifier(n_neighbors=7)

# Train the model using the scaled training sets
knnclf.fit(X_train_scaled, y_train)
y_pred = knnclf.predict(X_test_scaled)


# In[66]:


# Model Accuracy, how often is the classifier correct?
print(("Accuracy:",accuracy_score(y_test, y_pred)))


# ### That increases the accuracy a lot!

# # Support Vector Machine(SVM)

# In[67]:


from sklearn.svm import LinearSVC

svmclf = LinearSVC(C=50)
svmclf.fit(X_train, y_train)

print(('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(svmclf.score(X_train, y_train))))
print(('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(svmclf.score(X_test, y_test))))


# **Let's try on scaled data**

# In[68]:


svmclf = LinearSVC()
svmclf.fit(X_train_scaled, y_train)

print(('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(svmclf.score(X_train_scaled, y_train))))
print(('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(svmclf.score(X_test_scaled, y_test))))


# # Kernelize SVM
# 

# # Magic Weapon#2: **Support Vector Machine with RBF kernel**

# In[69]:


from sklearn.svm import SVC

svcclf = SVC(gamma=0.1)
svcclf.fit(X_train, y_train)

print(('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(svcclf.score(X_train, y_train))))
print(('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(svcclf.score(X_test, y_test))))


# **Look Accuracy on Training data, lol**

# In[70]:


svcclf = SVC(gamma=50)
svcclf.fit(X_train_scaled, y_train)

print(('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(svcclf.score(X_train_scaled, y_train))))
print(('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(svcclf.score(X_test_scaled, y_test))))


# # Decision Tree

# In[71]:


from sklearn.tree import DecisionTreeClassifier

dtclf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)

print(('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(dtclf.score(X_train, y_train))))
print(('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(dtclf.score(X_test, y_test))))


# Performed Well!

# # Random Forest
# 
# Secondly, I would like to introduce one of the most popular algorithms for classification (but also regression, etc), Random Forest! In a nutshell, Random Forest is an ensembling learning algorithm which combines decision trees in order to increase performance and avoid overfitting.

# In[72]:


from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier(random_state=2)


# # Magic Weapon #3: Hyperparameter Tuning

# Below we set the hyperparameter grid of values with 4 lists of values:
# 
# - 'criterion' : A function which measures the quality of a split.
# - 'n_estimators' : The number of trees of our random forest.
# - 'max_features' : The number of features to choose when looking for the best way of splitting.
# - 'max_depth' : the maximum depth of a decision tree.

# In[73]:


# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7]    
}


# In[74]:


from sklearn.model_selection import GridSearchCV

randomForest_CV = GridSearchCV(estimator = rfclf, param_grid = param_grid, cv = 5)
randomForest_CV.fit(X_train, y_train)


# Let's print our optimal hyperparameters set

# In[75]:


randomForest_CV.best_params_


# In[76]:


rf_clf = RandomForestClassifier(random_state = 2, criterion = 'gini', max_depth = 7, max_features = 'auto', n_estimators = 100)

rf_clf.fit(X_train, y_train)


# In[77]:


predictions = rf_clf.predict(X_test)


# In[78]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions) * 100


# Let's print our solutions

# # Magic Weapon #4: All model Accuracy Score

# In[79]:


#Linear Model
print(("Linear Model R-Squared for Train set: {:.3f}".format(linreg.score(X_train, y_train))))
print(("Linear Model R-Squared for test set: {:.3f}" .format(linreg.score(X_test, y_test))))
print()

#Logistic Regression
print(("Logistic Regression R-Squared for Train set: {:.3f}".format(logreg.score(X_train_scaled, y_train))))
print(("Logistic Regression R-Squared for test set: {:.3f}" .format(logreg.score(X_test_scaled, y_test))))
print()

#KNN Classifier
print(("KNN Classifier Accuracy:",accuracy_score(y_test, y_pred)))
print()

#SVM
print(('SVM Accuracy on training set: {:.2f}'
     .format(svmclf.score(X_train_scaled, y_train))))
print(('SVM Accuracy on test set: {:.2f}'
     .format(svmclf.score(X_test_scaled, y_test))))
print()

#Kerelize SVM
print(('SVC Accuracy on training set: {:.2f}'
     .format(svcclf.score(X_train_scaled, y_train))))
print(('Accuracy on test set: {:.2f}'
     .format(svcclf.score(X_test_scaled, y_test))))
print()

#Decision Tree
print(('Accuracy of Decision Tree on training set: {:.2f}'
     .format(dtclf.score(X_train, y_train))))
print(('Accuracy of Decision Tree on test set: {:.2f}'
     .format(dtclf.score(X_test, y_test))))
print()

#Random Forest
print(('Random Forest Accuracy:{:.3f}'.format(accuracy_score(y_test, predictions) * 100)))


# # Submitting the solutions
# 
# I am choosing SVC model for the instance, you can try submiting solution with different models

# In[80]:


scaler = MinMaxScaler()

train_conv = scaler.fit_transform(train.drop(['Survived', 'PassengerId'], axis=1))
test_conv = scaler.transform(test.drop(['PassengerId'], axis = 1))


# In[81]:


svcclf = SVC(gamma=50)
svcclf.fit(train_conv, train['Survived'])


# In[82]:


test['Survived'] = svcclf.predict(test_conv)


# In[83]:


test[['PassengerId', 'Survived']].to_csv('MySubmission1.csv', index = False)


# # Plz Upvote!
