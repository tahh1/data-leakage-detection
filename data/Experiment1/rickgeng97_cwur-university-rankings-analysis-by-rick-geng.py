#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Wranlging
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Classification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# # List of Questions we would like to answer
# Question 1: Identify Universities whose rank has been non-decreasing. <br>
# Question 2: Which factors are more determining at rankings <br>
# Question 3: Predict University Rankings based on different factors <br>

# ### But first, let's see some basic stats. 

# In[ ]:


rank = pd.read_csv('../input/cwurData.csv')


# In[ ]:


rank.info()


# In[ ]:


rank.describe()


# In[ ]:


list(rank.columns.values)


# In[ ]:


#Distinct Years
rank.year.unique()


# In[ ]:


#Count of Institution by country in 2015
ins_count = rank[rank['year'] == 2015].groupby('country').size().sort_values(ascending = False)
plt.figure(figsize = (15,15))
ax = sns.barplot(x = ins_count.values, y = ins_count.index)
ax.set(xlabel = 'Country', ylabel = 'Number of Institution')
for i in ax.patches:
    ax.text(i.get_width()+3.0, i.get_y()+0.6,i.get_width().astype(int), color='black', ha="center")
plt.xticks(rotation = 70)
plt.show()


# In[ ]:


#Number of top 100 institution by country in 2015
top_count = rank[rank['year'] == 2015].head(100).groupby('country').size().sort_values(ascending = False)
plt.figure(figsize = (15,10))
ax = sns.barplot(x = top_count.values, y = top_count.index)
ax.set(xlabel = 'Country', ylabel = 'Number of top 100 Institution')
for i in ax.patches:
    ax.text(i.get_width()+0.5, i.get_y()+0.6,i.get_width().astype(int))
plt.xticks(rotation = 70)
plt.show()


# In[ ]:


#Percentage of top 100 institution by country in 2015
per_count = top_count/ins_count
per_count.dropna(inplace = True)
per_count.sort_values(ascending = False, inplace = True)
plt.figure(figsize = (15,10))
ax = sns.barplot(x = per_count.values, y = per_count.index)
ax.set(xlabel = 'Country', ylabel = 'Percentage of top 100 Institution')
for i in ax.patches:
    ax.text(i.get_width(), i.get_y()+0.6,str(round(i.get_width()*100,1))+'%')
plt.xticks(rotation = 70)
plt.show()


# ## Question 1: Identify Universities whose rank has been non-decreasing.

# In[ ]:


#Define a new dataframe
institution = list(rank.institution.unique())
non_decreasing = pd.DataFrame(data=institution,columns=['institution'])
non_decreasing.head()


# In[ ]:


#filtering 
def non_decreasing_rank(institution):
    world_rank = list(rank[rank.institution == institution]['world_rank'])
    count = rank.groupby('institution').size()[institution]
    for i in range(1,count): #1,2,3
        if world_rank[i-1] < world_rank[i]:
            return False
    return True

#Moment of truth
non_decreasing[non_decreasing['institution'].apply(non_decreasing_rank) == True]


# ## Question 2 : Which factors are more determining at rankings

# In[ ]:


#We are using 2015 figures
rank2015 = rank[rank.year == 2015]
rank2015.drop(['country','national_rank','year','broad_impact'],axis = 1, inplace = True)
rank2015.head()


# In[ ]:


y = rank2015.quality_of_education.max() + 1


# In[ ]:


factor = list(rank2015.columns.values)[2:9]
factor


# In[ ]:


for i in range(len(factor)):
    z = rank2015[factor[i]].apply(lambda x:y-x)
    plt.figure(i)
    sns.regplot(x=z, y='score', data = rank2015)


# In[ ]:


cor = pd.DataFrame()
for i in range(len(factor)):
    cor[factor[i]] = rank2015[factor[i]].apply(lambda x:y-x)
cor['score'] = rank2015.score
cor.corr() 


# ## Question 3: Predict University Rankings based on different factors

# In[ ]:


score = rank.score
train = rank[factor] 
lab_enc = preprocessing.LabelEncoder()
score_encoded = lab_enc.fit_transform(score)


# In[ ]:


x_train, y_train, x_test, y_test = train_test_split(train,score_encoded,train_size = 0.9, random_state = 0)


# In[ ]:


#Decision Tree
tree = DecisionTreeClassifier()
tree.fit(x_train,x_test)
y_pred = tree.predict(y_train)
y1 = lab_enc.inverse_transform(y_test)
y2 = lab_enc.inverse_transform(y_pred)


# In[ ]:


np.corrcoef(y1,y2)
sns.regplot(y1,y2)


# It looks like the predicted values are very similar to true values. 

# In[ ]:


#Percentage of predicted score lies in between ±0.5 of the true score. 
fit = 0
for i in range(len(y1)):
    if (y1[i] - 0.5) <= y2[i] <= (y1[i] + 0.5):
        fit = fit + 1
        
print((fit/len(y1)))


# DecisionTree is useful for classifying discrete random variables or binned continous random variables. In the case of dealing with unbinned continous variables, it is actually not too bad. But, there must be better models.

# In[ ]:


#K-Nearest Neighbors
neigh = KNeighborsClassifier()
neigh.fit(x_train, x_test)
y_pred = neigh.predict(y_train)
y1 = lab_enc.inverse_transform(y_test)
y2 = lab_enc.inverse_transform(y_pred)


# In[ ]:


sns.regplot(y1,y2)
np.corrcoef(y1,y2)


# In[ ]:


#Percentage of predicted score lies in between ±0.5 of the true score. 
fit = 0
for i in range(len(y1)):
    if (y1[i] - 0.5) <= y2[i] <= (y1[i] + 0.5):
        fit = fit + 1
        
print((fit/len(y1)))


# In[ ]:


#Random Forest
forest = RandomForestClassifier()
forest.fit(x_train,x_test)
y_pred = forest.predict(y_train)
y1 = lab_enc.inverse_transform(y_test)
y2 = lab_enc.inverse_transform(y_pred)


# In[ ]:


sns.regplot(y1,y2)
np.corrcoef(y1,y2)


# Random Forest is the strongest classifier here

# In[ ]:


#Percentage of predicted score lies in between ±0.5 of the true score. 
fit = 0
for i in range(len(y1)):
    if (y1[i] - 0.5) <= y2[i] <= (y1[i] + 0.5):
        fit = fit + 1
        
print((fit/len(y1)))

