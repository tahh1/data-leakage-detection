#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# 
# 
# ## Introduction
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# ![The Grand Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# ## Problem Statement 
# In this challenge, we are tasked to complete the analysis of what kinds of people were likely to survive. (And before you ask, no, we are not going to solve the problem of whether Rose and Jack could have both survived on the door in Titanic).
# 
# ![They could not have both survived.](https://www.thesun.co.uk/wp-content/uploads/2017/01/nintchdbpict000268497624-e1485887027515.jpg?strip=all&w=960)
# 
# ## Approach
# Using data visualization techniques, we identify relationships between the various features in the dataset. From the visualizations, we create new ones which are good indicators of whether a passenger survived the tragedy. 
# 
# Next, we impute missing data based on these features which are highly correlated with the column with missing data.
# 
# After the data is cleaned, features are then selected from the existing pool. Following which, we employ various machine learning algorithms (such as Random Forest and Gradient Boosting) with hyperparameter turning and cross-validation to find the best model, and predict whether a passenger survived given his/her features across all the models based on our model.
# 
# We found that the Random Forest model with 1000 trees performed the best on the cross-validation datasets. In addition, it generalized pretty well to the testing dataset. As such, we used the Random Forest model to predict for the testing dataset.
# 
# **If only saving the Titanic could be as easy...**
# ![If only saving the Titanic could be so easy...](https://imgs.xkcd.com/comics/salvage.png)
# 
# 
# ## Evaluation
# Following [Kaggle's evaluation method](https://www.kaggle.com/c/titanic#evaluation), we will use the accuracy metric to evaluate our models.
# 
# ## Afternote
# Our model achieved a score of 0.80383. This was a huge improvement from my previous score, and places us at the top 15 percentile of the competition.
# 
# **Note**
# *  I cannot stress the importance of the [Pytanic Kernel](https://www.kaggle.com/headsortails/pytanic). The kernel has helped me to formulate and develop various hypotheses, identify and create key features which are correlated with our test label (whether a particular passenger survived the Titanic Tragedy). Please check Heads or Tails' kernel out; you won't regret it!

# ### Importing key libraries and reading dataframes
# 
# As usual, we begin by importing key libraries to read the data into Python, and allow us to conduct data analysis.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[3]:


combined = pd.concat([df_train, df_test])


# ### Exploratory Data Analysis and Feature Creation
# 
# First, we check the structure of the dataframe. How does the dataframe look like?

# In[4]:


combined.head()


# In[5]:


combined.apply(set) # Unique values that the features can take on


# First, we note that there are NA values in the dataframe. We may have to formulate a strategy to impute the missing data.
# 
# In addition, we also note that the column, Name, does not appear to be informative. Can we give meaning to the column by creating new features using this feature? Intuitively, we should be able to use the Name column to create 2 new columns - the First Name of the person and his/her Title.

# In[6]:


def first_name(x): 
    return x.split(', ')[0]

combined['FirstName'] = combined['Name'].apply(first_name)


# In[7]:


def title(x):
    return x.split(', ')[1].split('.')[0]

combined['Title'] = combined['Name'].apply(title)


# In[8]:


combined.head()


# Next, we take a quick look at the unique values, as well as the counts of the values, that the feature Title can take on.

# In[9]:


print((combined['Title'].value_counts()))
print((len(set(combined['Title']))))


# We note that there a total of 18 unique titles in the Title column (in both the training and the test set). That's too many. It appears that there are some 'misclassified' observations - for example, Ms, Mme and Mlle ought to be classified as Miss. From our observation, 4 titles dominate the distribution - Mr, Miss, Mrs and Master. Let's create a new category to classify the other Titles.

# In[10]:


def unique_title(x):
    if x in ('Mr', 'Miss', 'Mrs', 'Master'): return x
    elif x in ('Ms', 'Mlle', 'Mme'): return 'Miss'
    else: return 'Others'
    
combined['Title'] = combined['Title'].apply(unique_title)


# After sorting the Title column, let's take a quick look at the age distribution of the dataset.

# In[11]:


plt.figure(figsize=(15, 10))
plt.subplot(1, 1, 1)
plt.hist(combined['Age'].dropna(), bins=list(range(0, 81, 1)), color = 'green')
plt.title('Age Distribution of Combined Dataset')
plt.show()


# Judging from the Age distribution of the dataset, it appears that there were many young and middle-aged persons on the Titanic. Nothing really seems to stand out, apart from the fact that there were quite a few elderly people on board.
# 
# This begs the question: who are more likely to survive based on their age alone?

# In[12]:


plt.figure(figsize=(15, 10))
plt.subplot(1, 1, 1)
sns.distplot(combined[combined.Survived == 0].Age.dropna(), bins=list(range(0, 81, 1)), 
             color = 'red', label = 'Perished')
sns.distplot(combined[combined.Survived == 1].Age.dropna(), 
             bins=list(range(0, 81, 1)), color = 'blue', label = 'Survived')

plt.title('Age Distribution of Survivors')
plt.legend()
plt.show()


# As it turns out, children under the age of 10 are typically more likely to survive than perish. For all other age groups, the trend is reversed; people are more likely to perish than survive. Let's create a feature to capture this trend.

# In[13]:


combined['Child'] = (combined['Age'] <= 10)


# We now turn our attention to the Ticket type.

# In[14]:


print(('Total Unique Values for Ticket: ', len(set(combined['Ticket']))))
print(('Length of Combined Dataset: ', len(combined)))


# Similarly, the Ticket feature does not appear to be terribly informative as well. Let's take a quick look at the unique values in the column. There are 929 unique values out of a total 1309 values, suggesting that some of the tickets might be held by more than 1 person. What's the distribution of shared tickets? Let's find out.

# In[15]:


shared_tickets = combined.groupby('Ticket')['Name'].transform('count')

plt.figure(figsize=(15, 10))
plt.hist(shared_tickets.values)
plt.show()

combined['SharedTickets'] = shared_tickets.values


# As it turns out, many of the ticket holders travel alone. There are also large groups (8 or more people travelling together). Are solo travellers more likely to survive? Let's take a look.

# In[16]:


plt.figure(figsize=(20, 10))

sns.distplot(combined[(combined.Survived == 0)].SharedTickets.dropna(),
             bins = list(range(1, 12, 1)), kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Survived == 1)].SharedTickets.dropna(),
             bins = list(range(1, 12, 1)), kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.title('Distribution of Shared Tickets')
plt.legend()
plt.show()


# As it turns out, individuals with shared tickets (i.e. SharedTickets > 1) have a much higher likelihood of surviving. We proceed to convert this feature into a binary variable. However, this relationship does not hold true for all travellers. In particular, we note that this phenomena holds true for individuals who are sharing their tickets with 1 to 3 **other** passengers.

# In[17]:


def shared_tickets(x): 
    if x > 1: return(0)
    else: return(1)

def good_shared_tickets(x):
    if x > 1 and x < 5: return(1)
    else: return(0)

combined['GoodSharedTickets'] = combined['SharedTickets'].apply(good_shared_tickets)
combined['Alone'] = combined['SharedTickets'].apply(shared_tickets)


# Lastly, to clean up the Ticket Column, we rely on the same strategy that [Heads or Tails](https://www.kaggle.com/headsortails/pytanic) used; that is, we looked at the first character of the string to differentiate different types of tickets.

# In[18]:


combined['TicketType'] = combined['Ticket'].apply(lambda x: x[0])

print(('Unique Ticket Values: ', len(set(combined['Ticket']))))
print(('Unique Ticket Values (First Character): ', len(set(combined['TicketType']))))


# Using the first character of the ticket, we managed to reduce the ticket column from 929 unique values to 16.
# 
# Are there any missing values in the dataset? We can use the function ``` df.isnull().sum() ``` to find out.

# In[19]:


combined.isnull().sum()


# We note that there are missing values in the Age, Cabin, Embarked, Fare and Survived (not suprisingly at all, since the Survived variable is the variable we are trying to predict in the first place) columns. In particular, there are 263 missing Age values and 1014 missing Cabin values. Let's take a quick look at the unique values that the Cabin feature can take on.
# 
# There appears to be missing values in the Age, Fare, Cabin and Embarked features. In particular, the Cabin feature seems to be missing many values. Let's take a quick look at the unique values that the Cabin feature could take on.

# In[20]:


print((len(set(combined['Cabin']))))


# In[21]:


print((combined['Cabin'].value_counts()[0:5]))


# Looking at 5 unique values of the Cabin feature, 2 things stand out:
# 
# 1. The cabins typically start with an alphabet. The alphabet may be an indicator of where the cabin is located at (the Deck location), or it could simply imply the class of the cabin e.g. cabins that start with the letter A is better.
# 
# 2. There are several observations with more than 1 cabin. Typically, shared cabins have more than 1 person staying in them.
# 
# 3. The large amount of missing values in the Cabin column might be attributed to the fact that only the Survivors (which lived to tell the tale) reported their Cabin numbers.
# 
# With this knowledge, we could create 2 new features from the Cabin variable. The first feature we can create is a binary variable - it takes on value 1 if the individual has a Cabin to stay in, and 0 otherwise. Our second feature returns the type of cabin that the individual is staying in.

# In[22]:


def cabintostay(x):
    if pd.isnull(x): return(0)
    else: return(1)

combined['CabinToStay'] = combined['Cabin'].apply(cabintostay)


# In[23]:


def cabinclass(x):
    if pd.isnull(x): return('N')
    else: return (x.split())[0][0]

combined['CabinClass'] = combined['Cabin'].apply(cabinclass)


# Let's take a quick look at the summary of our dataset.

# In[24]:


combined.describe()


# Next. we take a closer look at the SibSp and Parch features. In particular, we hope to answer the question of whether individuals with more Siblings/Spouses or Parents/Children were more likely to survive. Let's take a look at this using a violin plot.

# In[25]:


plt.figure(figsize=(20,10))

plt.subplot(121)
sns.distplot(combined[combined.Survived == 0].Parch.dropna(), bins=list(range(0, 10, 1)), 
             kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[combined.Survived == 1].Parch.dropna(), bins=list(range(0, 10, 1)), 
             kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')
plt.ylim([0, 1])

plt.subplot(122)
sns.distplot(combined[combined.Survived == 0].SibSp.dropna(), bins=list(range(0, 10, 1)), 
             kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[combined.Survived == 1].SibSp.dropna(), bins=list(range(0, 10, 1)), 
             kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.ylim([0, 1])
plt.legend()
plt.show()


# Generally, individuals travelling with 1 or 2 Siblings/Spouses/Parents/Children were more likely to survive than their counterparts. Let's create a new feature, Family Size, that takes into account the total Parch and SibSp an individual has onboard the Titanic.

# In[26]:


combined['Family'] = combined['Parch'] + combined['SibSp']


# What is the family size distribution of individuals onboard the Titanic?

# In[27]:


plt.figure(figsize=(20,10))

sns.distplot(combined[combined.Survived == 0].Family.dropna(), bins=list(range(0, 12, 1)), 
             kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[combined.Survived == 1].Family.dropna(), bins=list(range(0, 12, 1)), 
             kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.show()


# As it turns out, families that are large, but not **that** large are more likely to survive the tragedy.

# In[28]:


combined['MiddleFam'] = ((combined['Family'] > 0) & (combined['Family'] < 4))


# Next, we turn to the question: does the area of embarkment or the fare you pay determine whether you survive?

# In[29]:


plt.figure(figsize=(20,8))

plt.subplot(1, 2, 1)
sns.distplot(combined[combined.Survived == 0].Fare.dropna(), bins = list(range(0, 500, 10)),
             norm_hist = True, color = 'red', label = 'Perished')
sns.distplot(combined[combined.Survived == 1].Fare.dropna(), bins = list(range(0, 500, 10)),
             norm_hist = True, color = 'blue', label = 'Survived')
plt.legend()

plt.subplot(1, 2, 2)
sns.barplot('Embarked', 'Survived', data = combined)

plt.show()


# From our simple plots, it does appear that those that pay higher fares are more likely to survive. In addition, we note that embarking at 'C' results in higher likelihood of surviving the tragedy. Also, we note that the lower bound of the 95% confidence interval around 'C' does not include 40%, indicating that embarkment point 'C' is indeed a potentially powerful predictor of survival, and is statistically significant.
# 
# Before we jump to conclusions, we have to understand that there might be confounding factors at play. For example, if more females than males boarded at 'C', and females were more likely to survive the tragedy, this implies that the place of embarkment does not play a significant role in predicting whether a particular passenger survived the tragedy. We have to bear these nuances in mind when conducting our data analysis.

# In[30]:


fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)

sns.boxplot('Embarked', 'Fare', data = combined, ax = ax1)

tab1 = pd.crosstab(combined['Embarked'], combined['Sex'])
plot1 = tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax2)

plt.show()


# As it turns out, passengers who embarked at 'C' tended to pay higher fares. However, there does not appear to be any other significant confounding factors, such as Gender composition, at play.
# 
# Do the effects on Mean Survival Rate based on higher fare prices differ between Males and Females?

# In[31]:


plt.figure(figsize=(20,10))

plt.subplot(121)
sns.distplot(combined[(combined.Survived == 0) & (combined.Sex == 'male')].Fare.dropna(), 
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Survived == 1) & (combined.Sex == 'male')].Fare.dropna(), 
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.ylim([0, 0.06])
plt.title('Fare Distribution of Survivors and Non-survivors [Males]')

plt.subplot(122)
sns.distplot(combined[(combined.Survived == 0) & (combined.Sex == 'female')].Fare.dropna(), 
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Survived == 1) & (combined.Sex == 'female')].Fare.dropna(), 
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.ylim([0, 0.06])
plt.title('Fare Distribution of Survivors and Non-survivors [Females]')

plt.show()

print(((combined[(combined.Survived == 0) & (combined.Sex == 'male')].Fare.mean()), 
       (combined[(combined.Survived == 1) & (combined.Sex == 'male')].Fare.mean())))
print(((combined[(combined.Survived == 0) & (combined.Sex == 'female')].Fare.mean()), 
       (combined[(combined.Survived == 1) & (combined.Sex == 'female')].Fare.mean())))


# As it turns out, Survivors tend to pay a higher price (no pun intended) for their fares. However, we do note that the probability of survival increases with the fare you pay, no matter which gender you are. However, the effects are asymmetric. For individuals who paid less than 10 dollars for their ticket, males are much more likely to die than females. The effect becomes more nuanced after. For femlaes, you are more likely to survive if you paid at least 50 dollars for your ticket.
# 
# One side note though: for males who paid more than 200 dollars, it turns out that most of them died, whereas the females who paid the same amount survived.

# In the graph, we observe that some tickets cost more than $500. Let's investigate this trend in more detail.

# In[32]:


print(('Number of Tickets costing more than $500: ', sum(combined.Fare.dropna() > 500)))


# In[33]:


combined[combined.Fare > 500] # What are the 4 rows?


# Looking closely at the 4 rows, something stands out. The passengers were all sharing tickets - PC17755. Also, we note that 3 out of the 4 individuals survived. We have previously identified that individuals that share tickets are more likely to die or survive together. In addition, I believe that fare prices were not adjusted for the number of members sharing the ticket.
# 
# To test this hypothesis, we focus on obtaining the fare distribution of individuals which have shared tickets, and compare them to those which have not. If it turns out that the fare distributions are systematically different, then our hypothesis is right.

# In[34]:


plt.figure(figsize=(20,8))

plt.subplot(121)
sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 0)].Fare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 1)].Fare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.title('Shared Tickets Fare Distribution')

plt.subplot(122)
sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 0)].Fare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 1)].Fare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.title('Normal Tickets Fare Distribution (Passengers who travelled alone)')

plt.show()


# Looking at the 2 distributions, they don't appear similar at all. Let's test this formally using a [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test) test.

# In[35]:


from scipy import stats

stats.ks_2samp(combined[(combined.Alone == 0) & (combined.Survived == 0)].Fare.dropna(),
               combined[(combined.Alone == 1) & (combined.Survived == 0)].Fare.dropna())


# With a pvalue of nearly 0, we reject the null hypothesis that the 2 distributions are the same, and can conclude that they were drawn from separate distributions. This tells us that we should normalize the distributions.

# In[36]:


shared_tickets = combined.groupby('Ticket')['Name'].transform('count')
combined['AdjustedFare'] = combined['Fare'] / shared_tickets


# Let's check how our distribution looks like after data imputation.

# In[37]:


plt.figure(figsize=(20,8))

plt.subplot(121)
sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 0)].AdjustedFare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Alone == 0) & (combined.Survived == 1)].AdjustedFare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.title('Shared Tickets Adjusted Fare Distribution')

plt.subplot(122)
sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 0)].AdjustedFare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Alone == 1) & (combined.Survived == 1)].AdjustedFare.dropna(),
             bins=list(range(0, 500, 10)),  kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.title('Normal Tickets Adjusted Fare Distribution (Passengers who travelled alone)')

plt.show()


# Let's look at the general distribution.

# In[38]:


plt.figure(figsize=(20,8))
sns.distplot(combined[(combined.Survived == 0)].AdjustedFare.dropna(),
             bins=list(range(0, 150, 1)), kde = False, norm_hist = True, 
             color = 'red', label = 'Perished')
sns.distplot(combined[(combined.Survived == 1)].AdjustedFare.dropna(),
             bins=list(range(0, 150, 1)), kde = False, norm_hist = True, 
             color = 'blue', label = 'Survived')

plt.legend()
plt.title('Adjusted Fare Distribution')
plt.show()


# Following our adjustment, the relationship between fares and survival appears to be much more convincing now. We note that for fares which are between 0 - 10, the probability of perishing is greater than the probability of surviving. Similar to [Heads or Tails](https://www.kaggle.com/headsortails/pytanic), we use the threshold of 10 to discern between pricey and cheap tickets.

# In[39]:


combined['CheapTickets'] = combined['AdjustedFare'] <= 10


# In[40]:


plt.figure(figsize=(20,8))
sns.pointplot('Sex', 'Survived', hue = 'Pclass', data = combined)
plt.show()


# The likelihood of a Female surviving is higher than that of a comparable male across all classes. However, the 'survival premium' associated with being a Female is asymmetric; it is different across different classes. The slopes of the different lines represent the 'survival premium' associated with being Female.
# 
# It appears that the 'survival premium' is the greatest for Class 2 passengers i.e. Class 2 passenger females have the highest 'survival premium', compared to females in the other classes.

# In[41]:


fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)

p = sns.factorplot('Embarked', 'Survived', hue = 'Sex', data = combined, ax = ax1)
plt.close(p.fig)

g = sns.factorplot('Embarked', 'Survived', hue = 'Pclass', data = combined, ax = ax2)
plt.close(g.fig)


# Does the 'survival premium' change across different embarkment points? It doesn't seem to be the case. From the first plot, we observe that females stand a larger chance of surviving across all embarkment points. In addition, we note that individuals that board at 'C' stands the largest chance of surviving the tragedy, while individuals which boarded Titanic at 'Q' stands the lowest chance (albeit significant uncertainty, perhaps due to the small observations).
# 
# Was the difference in survival probability across the different embarkment point due to the survival probability of each individual passenger class? It does not appear to be the case. From the second plot, we observe that there isn't any significant difference between survival probability across the 3 classes controlled for embarkment point (save for port 'C'). 
# 
# This suggests that the gender and passenger class composition of the passengers across the different embarkment points are the main drivers for the difference in the survival probability of different embarkment points.

# In[42]:


fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)

tab1 = pd.crosstab(combined['Embarked'], combined['Sex'])
plot1 = tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax1)
plot1 = plt.xlabel('Embarkation Point')
plot1 = plt.ylabel('Percentage')
plot1 = plt.title('Sex')

tab2 = pd.crosstab(combined['Embarked'], combined['Pclass'])
plot2 = tab2.div(tab2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax2)
plot2 = plt.xlabel('Embarkation Point')
plot2 = plt.ylabel('Percentage')
plot2 = plt.title('Pclass')

tab3 = pd.crosstab(combined['Embarked'], combined['GoodSharedTickets'])
plot3 = tab3.div(tab3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax3)
plot3 = plt.xlabel('Embarkation Point')
plot3 = plt.ylabel('Percentage')
plot3 = plt.title('Good Shared Tickets')

tab4 = pd.crosstab(combined['Embarked'], combined['CabinToStay'])
plot4 = tab4.div(tab3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax4)
plot4 = plt.xlabel('Embarkation Point')
plot4 = plt.ylabel('Percentage')
plot4 = plt.title('Cabin To Stay')


# It turns out that we were (partially) right. Compared to other embarkation points, 'C' has the largest percentage of Class 1 passengers. At the same time, most passengers which embarked at 'C' had cabins to stay in. However, we note that although 'Q' has a relatively large proportion of female passengers, it has the lowest survival probability out of the 3 embarkation points. This might be attributed to its huge proportion of Class 3 passengers, and ticketholders embarking at Q are typically not holding 'GoodSharedTickets', and do not have a Cabin to stay in.
# 
# This means that the embarkation point may or may not be a good predictor of whether an individual survived the tragedy, as it is correlated with the Pclass and Sex features.

# Finally, we ask whether the Cabin features are good predictors of whether a person has survived the tragedy. First, we check whether the Cabin features are correlated with Passenger Class and Fares. Intuitively, we expect tickets with cabins assigned should cost more.

# In[43]:


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)

g1 = sns.factorplot('CabinToStay', 'Survived', hue = 'Pclass', data = combined, ax = ax1)
plt.close(g1.fig)
sns.violinplot('CabinToStay', 'CheapTickets', hue = 'Survived', 
               data = combined, ax = ax2, split = True)

tab1 = pd.crosstab(combined['CabinToStay'].dropna(), 
                   combined[pd.notnull(combined.CabinToStay)].Sex)
tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax3)

tab2 = pd.crosstab(combined['CabinToStay'].dropna(), 
                   combined[pd.notnull(combined.CabinToStay)].Pclass)
tab2.div(tab2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax4)

plt.show()


# It appears that the high probability of survival for passengers who stayed in a cabin could be attributed to their passenger class, and their gender. Passengers who were staying in cabins were more likely to survive precisely because females and rich passengers were more likely to survive the tragedy.

# Finally, we turn to Ticket Types and Cabin Classes. Which ticket types or cabin classes are likely to result in higher survival probabilities?

# In[44]:


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)

g1 = sns.factorplot('CabinClass', 'Survived', 
                    kind = 'bar', data = combined, ax = ax1)
plt.close(g1.fig)
g2 = sns.factorplot('TicketType', 'Survived', 
                    kind = 'bar', data = combined, ax = ax2)
plt.close(g2.fig)
plt.show()


# From the factor plots, we observe that certain CabinClasses (Such as CabinClass B, C, D and E) and TicketTypes are associated with higher survival probabilities. Are confounding factors at play?

# In[45]:


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)

tab1 = pd.crosstab(combined['CabinClass'], 
                   combined['Sex'])
dummy1 = tab1.div(tab1.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax1)
dummy1 = plt.xlabel('Cabin Class')
dummy1 = plt.ylabel('Percentage')

tab2 = pd.crosstab(combined['CabinClass'], 
                   combined['Pclass'])
dummy2 = tab2.div(tab2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax2)
dummy2 = plt.xlabel('Cabin Class')
dummy2 = plt.ylabel('Percentage')

tab3 = pd.crosstab(combined['TicketType'], 
                   combined['Sex'])
dummy3 = tab3.div(tab3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax3)
dummy3 = plt.xlabel('Ticket Type')
dummy3 = plt.ylabel('Percentage')

tab4 = pd.crosstab(combined['TicketType'], 
                   combined['Pclass'])
dummy4 = tab4.div(tab4.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax = ax4)
dummy4 = plt.xlabel('Ticket Type')
dummy4 = plt.ylabel('Percentage')

plt.show()


# From our initial plots, we observe that Cabin Classes B, C, D and E have much higher survival probabilities. Looking at the crosstab plots, it becomes clear that they were predominantly female and rich passengers, traits which yielded higher survival probabilities. To see this more closely, we can take a look at the 'A' cabins. While passengers were predominantly from Passenger Class 1, these cabins had questionably low survival rates as they were dominated by male inhabitants. Nontheless, let's try to encapsulate this set of information in our prediction model.
# 
# For ticket types, we observe that ticketholders holding ticket types P, 1, 2 and 9 had higher survival probabilities than other ticketholders. This trend is a little bit surprising, given that ticketholders of these ticket types were predominantly male (save for ticket type 9 and maybe P), although they were, on average, richer than the other ticketholders.
# 
# After this analysis, we understand that the TicketType and CabinClass features could potentially be important in helping us predict whether a passenger survived. However, the selection criteria is key. Using the mean survival threshold of 0.4 for TicketType (which gives 5 groups in total) and 0.55 for Cabin Class (which also gives 5 groups in total), we attempt to filter key ticket types and cabin types which should, in theory, help us to predict the survival probability of the passengers.

# In[46]:


tickettype_groupby = combined.groupby('TicketType')['Survived'].apply(np.mean)

def good_ticket(ticket_type):
    if tickettype_groupby[ticket_type] > 0.4: return(1)
    else: return(0)
    
combined['GoodTicket'] = combined['TicketType'].apply(good_ticket)


# In[47]:


cabinclass_groupby = combined.groupby('CabinClass')['Survived'].apply(np.mean)

def good_cabin(cabinclass):
    if cabinclass_groupby[cabinclass] > 0.55: return(1)
    else: return(0)
    
combined['GoodCabinClass'] = combined['CabinClass'].apply(good_cabin)


# ### Imputing Missing Values
# 
# After our exploratory data analysis, we (finally) turn to impute the missing values in the dataset.
# 
# We first split our dataset into the training and testing dataset before the imputation.

# In[48]:


train = combined[:len(df_train)]
test = combined[len(df_train):]


# In[49]:


combined.isnull().sum()


# Noting that the missing Cabin values shouldn't pose any major problems, we can (**finally**) focus our attention on imputing the missing Age, Embarkation and Fare values.

# #### Imputing Missing Age
# 
# To impute the missing values for the column Age, we should rely on indicators which are correlated with Age to ensure that we are encapsulating all available information in our imputation. We plot a heatmap to identify these features.

# In[50]:


plt.figure(figsize=(30, 12))

plt.subplot(211)
train_male = train[train.Sex == 'male']
sns.heatmap(train_male.corr(), annot = True)
plt.title('Correlation Plot for Males')

plt.subplot(212)
train_female = train[train.Sex == 'female']
sns.heatmap(train_female.corr(), annot = True)
plt.title('Correlation Plot for Females')

plt.show()


# We rely on the Title feature in our imputation of the Age variable.

# In[51]:


age_prior = combined['Age'].copy()
groupby_impute_age = train.groupby(['Title']).apply(np.mean)[['Age']]
groupby_impute_age


# We return the mean age based on the passenger's Title.

# In[52]:


def return_age(age, title):
    if pd.notnull(age): return age
    else: return groupby_impute_age.ix[title]

return_age_vec = np.vectorize(return_age)
train['Age'] = return_age_vec(train['Age'], train['Title'])
test['Age'] = return_age_vec(test['Age'], test['Title'])


# In[53]:


plt.figure(figsize=(15, 10))
plt.subplot(1, 1, 1)

sns.distplot(combined['Age'], bins=list(range(0, 81, 1)), color = 'orange')
sns.distplot(age_prior.dropna(), bins=list(range(0, 81, 1)), color = 'green')
plt.title('Age Distribution of Combined Dataset')

plt.show()


# After our imputation, there appears to be a large increase in the proportion of passengers aged 20 and 40, at the expense of other age groups.

# We have 3 more missing values to fill - 2 missing embarkation points and 1 missing Fare. We can rely on the correlation plot to help us find the best variables to predict these features.
# 
# For point of embarkation, it turns out that previously, we have identified the Passenger Class featur.
# 
# For Fare, it turns out that Passenger Class and CabinToStay are good variables that we can rely on to predict how much the passenger paid.

# In[54]:


train[train.Embarked.isnull()]


# As it turns out, the 2 observations without an embarkation point were Class 1 Passengers, Females, had shared tickets and a Cabin to stay. Looking at our previous crosstab plots, they appeared to embark at point C.

# In[55]:


train.ix[61, 'Embarked'] = 'C'
train.ix[829, 'Embarked'] = 'C'


# Lastly, we impute the missing fare. Let's take a quick look at the row.

# In[56]:


test[test.Fare.isnull()]


# As a Class 3 Passenger with no Cabin to stay and the point of embarkation at 'S', let's find the mean fare of such a passenger.

# In[57]:


print((combined.groupby(['Pclass', 'CabinToStay', 
                        'Embarked'])['Fare'].apply(np.mean).ix[3].ix[0].ix['S']))


# The average fare for such a passenger is 14.52. Let's impute it in, along with the other missing Fare features.

# In[58]:


test.ix[152, 'Fare'] = combined.groupby(['Pclass', 'CabinToStay', 
                                         'Embarked'])['Fare'].apply(np.mean).loc[3].loc[0].loc['S']


# In[59]:


# Adjusted Fare is the same as Fare since passenger is alone
test.ix[152, 'AdjustedFare'] = test.ix[152, 'Fare']
test.ix[152, 'CheapTickets'] = test.ix[152, 'AdjustedFare'] <= 0


# ### Feature Processing and Encoding
# 
# So far, we have yet to convert and encode some of our features. Let's do so before we employ our machine learning algorithms to predict whether a particular passenger survived.

# Let's take a look at the correlation plot after we have created these new features.

# In[60]:


plt.figure(figsize=(20, 12))
sns.heatmap(combined[:len(df_train)].corr(), annot = True)
plt.show()


# Now that we have converted our features to integer variables, let's proceed to encode some of our key features, which will be used in our machine learning algorithms!

# In[61]:


from sklearn.preprocessing import LabelEncoder
encodeFeatures = ['Title', 'Pclass', 'Sex', 'Child', 'SharedTickets', 'Alone', 'Embarked',
                  'Family',  'GoodCabinClass', 'GoodTicket', 'CheapTickets', 'MiddleFam']

for i in encodeFeatures:
    combined[i] = combined[i].astype('category')
    
le = LabelEncoder()
combined_processed = combined[encodeFeatures].apply(le.fit_transform)


# ### Splitting into Subtraining, Subtesting and Testing datasets
# 
# As good data scientists, we split our training and testing datasets into Training, Cross-Validation and Testing data to evaluate our model. Given that we are dealing with a binary classification problem, we will employ the following techniques, along with hyperparameter tuning (aka GridSearch) with cross validation, to find the best prediction model:
# 
# 1. Support Vector Machine (with linear kernel)
# 2. Random Forest Classifier
# 3. Gradient Boosting Classifier
# 4. K-Nearest Neighbors

# In[62]:


# Splitting Training Data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X = combined_processed
y = combined['Survived']

X_test = X[len(train):].copy()
X_train = X[:len(train)].copy(); y_train = y[:len(train)].copy()

X_subtrain, X_subtest, y_subtrain, y_subtest = train_test_split(X_train, y_train, 
                                                                test_size = 0.2,
                                                                random_state = 42)


# ### Feature Selection and Model Fitting
# 
# We proceed to select key fetaures using Features Importances. Following which, we then fit our model using these features, and evaluate the models using the metric, Accuracy.

# In[63]:


# Set Random State
random_state = 1212


# #### Feature Importances

# In[64]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

rf = RandomForestClassifier(n_estimators = 500, random_state = random_state).fit(X_subtrain, y_subtrain)
gb = GradientBoostingClassifier(n_estimators = 300, random_state = random_state).fit(X_subtrain, y_subtrain)


# In[65]:


feature_importance = pd.DataFrame()
feature_importance['Features'] = X_subtrain.columns
feature_importance['RandomForest'] = rf.fit(X_subtrain, y_subtrain).feature_importances_
feature_importance['GBM'] = gb.fit(X_subtrain, y_subtrain).feature_importances_
feature_importance


# As it turns out, the features that we have selected are doing pretty well. None of them appears to be poor indicators of whether a passenger survived the tragedy. Let's use these features to fit our models.

# #### Model Fitting

# We will attempt to fit 4 binary classification model, with hyperparameter tuning using GridSearch. Based on the cross validation score and the accuracy metric, we will choose the model with the highest cross-validation and testing score, as it will probably generalize well to the test dataset. The models are:
# 
# * Support Vector Machine
# * Random Forest Classifier
# * Gradient Boosting Classifier
# * K-Nearest Neighbors

# In[66]:


# Model 1: Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC(random_state = random_state, probability = True)
param_grid = {'kernel': ['linear', 'rbf'],
              'C': np.logspace(0, 2, 20)}
svm_clf = GridSearchCV(svm, param_grid).fit(X_subtrain, y_subtrain)

svm_score = cross_val_score(svm_clf, X_subtrain, y_subtrain, cv = 5).mean()
print(svm_score)


# As it turns out, the Support Vector Machine scored a total of 0.8259!

# In[67]:


# Model 2: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 1000,
                            min_samples_split=10,
                            random_state = random_state)

param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [4, 5, 6]}
rf_clf = GridSearchCV(rf, param_grid).fit(X_subtrain, y_subtrain)

rf_score = cross_val_score(rf_clf, X_subtrain, y_subtrain, cv = 5).mean()
print(rf_score)


# Our Random Forest Model was able to achieve a score 0.8343, the highest so far!

# In[68]:


# Model 3: Gradient Boosting Model
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators = 500, 
                                max_depth = 1, random_state = random_state)

param_grid = {'learning_rate': np.logspace(-2, 2, 10),
              'loss': ['deviance', 'exponential']}
gb_clf = GridSearchCV(gb, param_grid).fit(X_subtrain, y_subtrain)

gb_score = cross_val_score(gb_clf, X_subtrain, y_subtrain, cv = 5).mean()
print(gb_score)


# The Gradient Boosting Classifier scored an accuracy of 0.8259, just slightly lower than the Random Forest Model!

# In[69]:


# Model 4: K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 10],
              'weights': ['uniform', 'distance']}
knn_clf = GridSearchCV(knn, param_grid).fit(X_subtrain, y_subtrain)

knn_score = cross_val_score(knn_clf, X_subtrain, y_subtrain, cv = 5).mean()
print(knn_score)


# The K-Nearest Neighbors model scored 0.8047.

# From the cross-validation scores of the 4 models, it does appear that the Random Forest model scored the highest amongst all the models. Let's see how well the Random Forest model generalizes to data not seen before i.e the testing set. 
# 
# If it turns out to generalize pretty well to the testing dataset, I think we have a winner here.

# In[70]:


print(('Training Score - SVM: ', svm_score))
print(('Testing Score - SVM: ',svm_clf.score(X_subtest, y_subtest)))

print('\n')

print(('Training Score - Random Forest Classifier: ', rf_score))
print(('Testing Score - Random Forest: ', rf_clf.score(X_subtest, y_subtest)))

print('\n')

print(('Training Score - Gradient Boosting Classifier: ', gb_score))
print(('Testing Score - Gradient Boosting Classifier: ', gb_clf.score(X_subtest, y_subtest)))

print('\n')

print(('Training Score - K-Nearest Neighbors Classifier: ', knn_score))
print(('Testing Score - K-Nearest Neighbors Classifier: ', knn_clf.score(X_subtest, y_subtest)))


# The Random Forest Classifier turns out to generalize pretty well to the testing dataset, scoring 0.82123 on the testing data. Although, the Gradient Boosting model did really on both the cross-validation and the testing dataset too (especially the testing set), it did not perform as well on the cross-validation dataset, scoring nearly 1% lower than the Random Forest Model.

# #### Ensembling
# 
# Before we use the Random Forest model to predict for the test set, we could also create an ensembling model using the predictions from the 4 models. In this case, we focus on using the predictions from the 3 models  - Random Forest, SVM and Gradient Boosting model as they have the highest cross validation score. Intuitively, this might make our model more robust. 
# 
# Before that, let's check for the correlation of the prediction on the subtesting datasets across the different models. If it turns out that the correlation between the different predictions is high, then we shouldn't expect our model to be much more robust than a standalone model. (**Why?**)

# In[71]:


cv = pd.DataFrame()
cv['SVM'] = svm_clf.predict(X_subtrain)
cv['Random Forest'] = rf_clf.predict(X_subtrain)
cv['Gradient Boosting'] = gb_clf.predict(X_subtrain)
cv['K-Nearest Neighbors'] = knn_clf.predict(X_subtrain)


# In[72]:


plt.figure(figsize=(20, 15))
sns.heatmap(cv.corr(), annot = True)
plt.show()


# As it turns out, the predictions from the 4 models are all pretty strongly correlated with one another (correlation are all larger than 0.70). This is a red flag. Nonetheless, we can still create an ensemble model, and see how well it generalizes to the testing dataset.
# 
# If we were to focus on the 3 top models (i.e. SVM, Random Forest and Gradient Boosting), the minimum correlation between the 3 models are 0.82!

# In[73]:


cv_score = pd.DataFrame(index=['Max Cross-Validation Score', 'Testing Score'])
cv_score['SVM'] = [svm_score, svm_clf.score(X_subtest, y_subtest)]
cv_score['Random Forest'] = [rf_score, rf_clf.score(X_subtest, y_subtest)]
cv_score['Gradient Boosting'] = [gb_score, gb_clf.score(X_subtest, y_subtest)]


# In[74]:


no_of_models = list(range(1, len(cv_score.columns) + 1))

plt.figure(figsize = (20, 8))

plt.plot(no_of_models, cv_score.T['Max Cross-Validation Score'], 
         no_of_models, cv_score.T['Testing Score'])

plt.legend(['Mean Cross-Validation Score', 'Testing Score'], loc='lower right')

plt.xticks(no_of_models, cv_score.T.index)
plt.ylim([0, 1])
plt.show()


# The cross-validation scores are similar to the testing score. Let's use an ensemble of the 3 models to predict for the testing data set.

# Let's create a new function to take into account the prediction of the 4 models.

# In[75]:


cv_pred = pd.DataFrame()
cv_pred['SVM'] = svm_clf.predict(X_subtrain)
cv_pred['Random Forest'] = rf_clf.predict(X_subtrain)
cv_pred['Gradient Boosting'] = gb_clf.predict(X_subtrain)


# In[76]:


from sklearn.metrics import accuracy_score

def pred(x):
    if x >= 2: return 1
    else: return 0
    
print((accuracy_score(y_subtrain, list(map(pred, cv_pred.sum(axis = 1))))))


# Using our ensemble method, we scored 0.8483 on our training set. What about our testing set?

# In[77]:


cv_pred = pd.DataFrame()
cv_pred['SVM'] = svm_clf.predict(X_subtest)
cv_pred['Random Forest'] = rf_clf.predict(X_subtest)
cv_pred['Gradient Boosting'] = gb_clf.predict(X_subtest)


# In[78]:


from sklearn.metrics import accuracy_score
ensemble = list(map(pred, cv_pred.sum(axis = 1)))
print((accuracy_score(y_subtest, cv_pred['Random Forest'])))


# Using a simple ensemble model, we were able to achieve a testing score of 0.82123, similar to that of the Random Forest model. It appears that the ensemble did not help to improve our cross-validation score significantly. Furthermore, we did not conduct cross-validation on the ensemble model. Due to these factors, let's use the Random Forest model to predict for the actual testing dataset.

# In[79]:


pred = rf_clf.predict(X_test)


# As a final sense-check, let's check to ensure that the mean survival rate that we have obtained is not too far away from the average survival rate in the training datasets.

# In[80]:


print(('Mean Survival Rate for Training Dataset: ', np.mean(df_train['Survived'])))
print(('Mean Survival Rate for Testing Dataset: ', np.mean(pred)))


# The training dataset had a mean survival rate of 0.384 while our prediction yields a mean survival rate of 0.340 on the testing set. These figures do not look too different (a 5% difference is still fine, I think), and we can submit our predictions to Kaggle.

# In[81]:


submission = pd.read_csv('../input/genderclassmodel.csv')
submission['Survived'] = list(map(int, pred))
submission.head()


# In[82]:


submission.to_csv('submission.csv', index = False)


# Now, let's submit it to Kaggle!
# 
# [Afternote: Our submission earned us a score of 0.80383, placing us at the top 15 percentile of the Kaggle competition. This was a marked improvement from my previous score, which only came in at 0.78947.]
