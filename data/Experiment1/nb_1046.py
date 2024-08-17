#!/usr/bin/env python
# coding: utf-8

# # Applied Data Science Capstone Project
# ## Accident Severity Prediction
# ***
# ### Table of Contents
# + Introduction : Business Problem
# + Data
# + Modeling
# + Model Evaluation
# + Results and Discussion
# 
# ### Introduction/Business Problem 
# 
# Road accidents are extremely common and they often lead to loss of property and even life. Hence its good to have a tool that can alert the drivers to be more careful depending on the weather and road conditions. If the severity is high the driver can decide whether to be extra cautious or delay the trip if possible.
# This tool can also help the police to enforce more safety protocols.
# 
# The goal of this project is to predict road accident severity depending on certain weather and road conditions and time of the day.
# The data set used for training the model is the one recorded by the Seattle Department of Transportation(SDOT) which includes all types of collisions from 2004 to present.
# It has around 194673 records with 38 attributes.
# 
# ### Data
# 
# We will be using the shared data, ie. the collision data recorded by the Seattle Department of Transportation(SDOT) which is avialable at - 
# https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv
# 
# 
# Inorder to develop a Accident Severity Predicting Model, we will be considering the following Attributes.
# 
# + WEATHER - A description of the weather conditions during the time of the collision.
# + ROADCOND - The condition of the road during the collision.
# + LIGHTCOND - The light conditions during the collision.
# 
# 
# The target is the Severity of collision which is represented by column :
# 
# + SEVERITYCODE - A code that corresponds to the severity of the collision
# 
# We have two possible outcomes for this in our data set :
# 1 - Property Damage Only Collision
# 2 - Injury Collision
# 

# In[1]:


#import required libraries
import pandas as pd
import numpy as np


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter
mpl.style.use('ggplot')


# #### Data Collection

# In[3]:


#data file - shared data for SDOT 
data_file = "https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv"


# In[4]:


#read data from file to pandas data frame
df = pd.read_csv(data_file)
df.head()


# #### Data Understanding

# In[5]:


df.shape


# In[6]:


#Checking the data types
df.dtypes


# In[7]:


df["SEVERITYCODE"].value_counts()


# In[8]:


df["SEVERITYDESC"].value_counts()


# #### Data Preparation and Pre-processing

# In[9]:


#Creating a new df with the independnet variables(attributes) and target variable
df_final = df[['SEVERITYCODE', 'WEATHER', 'ROADCOND','LIGHTCOND']].copy()
df_final.head()


# In[10]:


#Check for missing data
print("Missing values in each columns")
print(("SEVERITYCODE : " , df_final['SEVERITYCODE'].isnull().sum(axis=0)))
print(("WEATHER : " , df_final['WEATHER'].isnull().sum(axis=0)))
print(("ROADCOND : " , df_final['ROADCOND'].isnull().sum(axis=0)))
print(("LIGHTCOND : " , df_final['LIGHTCOND'].isnull().sum(axis=0)))


# In[11]:


#Since then no. rows with missing values is less compared to total no. of records, we can drop these rows
df_final.dropna(subset=['WEATHER', 'ROADCOND','LIGHTCOND'], axis=0, inplace=True)
df_final.shape


# In[12]:


#Analysing the values of Attribute - WEATHER
df_final.groupby(['WEATHER'])['SEVERITYCODE'].value_counts()


# In[13]:


#Weather conditions
plt.rcParams["figure.figsize"] = (8,6)
x = Counter(df_final["WEATHER"])
y = list(range(len(list(x.values()))))
plt.title("Weather Condition")
plt.ylabel("Number of Accidents")
plt.bar(y, list(x.values()))
plt.xticks(y, list(x.keys()), rotation='vertical')


# In[14]:


#Drop Unknown, Other and less impacting values
df_final.drop(df_final[df_final.WEATHER == 'Unknown'].index, inplace=True)
df_final.drop(df_final[df_final.WEATHER == 'Other'].index, inplace=True)
df_final.drop(df_final[df_final.WEATHER == 'Fog/Smog/Smoke'].index, inplace=True)
df_final.drop(df_final[df_final.WEATHER == 'Sleet/Hail/Freezing Rain'].index, inplace=True)
df_final.drop(df_final[df_final.WEATHER == 'Blowing Sand/Dirt'].index, inplace=True)
df_final.drop(df_final[df_final.WEATHER == 'Severe Crosswind'].index, inplace=True)
df_final.drop(df_final[df_final.WEATHER == 'Partly Cloudy'].index, inplace=True)


# In[15]:


#Analysing the values of Attribute - ROADCOND
df_final.groupby(['ROADCOND'])['SEVERITYCODE'].value_counts()


# In[16]:


#Road conditions
plt.rcParams["figure.figsize"] = (8,6)
x = Counter(df_final["ROADCOND"])
y = list(range(len(list(x.values()))))
plt.title("Road Condition")
plt.ylabel("Number of Accidents")
plt.bar(y, list(x.values()))
plt.xticks(y, list(x.keys()), rotation='vertical')


# In[17]:


#Drop Unknown, Other and less impactful values
df_final.drop(df_final[df_final.ROADCOND == 'Unknown'].index, inplace=True)
df_final.drop(df_final[df_final.ROADCOND == 'Other'].index, inplace=True)
df_final.drop(df_final[df_final.ROADCOND == 'Sand/Mud/Dirt'].index, inplace=True)
df_final.drop(df_final[df_final.ROADCOND == 'Standing Water'].index, inplace=True)
df_final.drop(df_final[df_final.ROADCOND == 'Oil'].index, inplace=True)


# In[18]:


#Analysing the values of Attribute - LIGHTCOND
df_final.groupby(['LIGHTCOND'])['SEVERITYCODE'].value_counts()


# In[19]:


#Light conditions
plt.rcParams["figure.figsize"] = (8,6)
x = Counter(df_final["LIGHTCOND"])
y = list(range(len(list(x.values()))))
plt.title("Light Condition")
plt.ylabel("Number of Accidents")
plt.bar(y, list(x.values()))
plt.xticks(y, list(x.keys()), rotation='vertical')


# In[20]:


#Drop Unknown and Other
df_final.drop(df_final[df_final.LIGHTCOND == 'Unknown'].index, inplace=True)
df_final.drop(df_final[df_final.LIGHTCOND == 'Other'].index, inplace=True)
df_final.drop(df_final[df_final.LIGHTCOND == 'Dark - No Street Lights'].index, inplace=True)
df_final.drop(df_final[df_final.LIGHTCOND == 'Dark - Street Lights Off'].index, inplace=True)
df_final.drop(df_final[df_final.LIGHTCOND == 'Dark - Unknown Lighting'].index, inplace=True)


# In[21]:


df_final.shape


# In[22]:


df_final.head(10)


# In[23]:


#Use one hot encoding technique to convert categorical varables to binary variables and append them to the feature Data Frame
#df_feature = pd.concat([df_final,pd.get_dummies(df_final[['WEATHER','ROADCOND','LIGHTCOND']])], axis=1)
wdummies = pd.get_dummies(df_final["WEATHER"])
rdummies = pd.get_dummies(df_final["ROADCOND"])
ldummies = pd.get_dummies(df_final["LIGHTCOND"])

#Merging with existing Dara Frame
df_final = df_final.join(wdummies)
df_final = df_final.join(rdummies)
df_final = df_final.join(ldummies)


# In[24]:


df_feature = df_final.copy()
df_feature = df_feature.drop(['SEVERITYCODE','WEATHER','ROADCOND','LIGHTCOND'],axis=1)
df_feature.head(10)


# In[25]:


df_feature.shape


# **After Data Cleaning and PreProcessing, we have a 12 features/attributes.**

# In[26]:


X = df_feature.copy()
X[0:5]


# In[28]:


y = df_final['SEVERITYCODE'].values
y[0:5]


# ### Modelling

# In[29]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[30]:


#split data into tarin and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print(('Train set:', X_train.shape,  y_train.shape))
print(('Test set:', X_test.shape,  y_test.shape))


# In[31]:


#Normalize Data
#Data Standardization give data zero mean and unit variance 
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)


# #### K Nearest Neighbour

# In[32]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[33]:


Ks = 10
mean_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
mean_acc


# In[34]:


print(( "The best accuracy was with", mean_acc.max(), "with k= ", mean_acc.argmax()+1)) 
k = mean_acc.argmax()+1
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)


# #### Decision Tree

# In[35]:


from sklearn.tree import DecisionTreeClassifier
dTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
dTree.fit(X_train,y_train)


# #### Logistic Regression

# In[36]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear')
LR.fit(X,y)


# ### Model Evaluation

# In[37]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# #### K Nearest Neighbour

# In[38]:


yhat_neigh = neigh.predict(X_test)


# In[40]:


print(("Test set Accuracy : ", metrics.accuracy_score(y_test, yhat_neigh)))
print(("Jaccard Similarity Score : ", jaccard_similarity_score(y_test, yhat_neigh)))
print(("F1  Accuracy : ", f1_score(y_test, yhat_neigh, average='weighted')))


# #### Decision Tree

# In[41]:


yhat_dtree = dTree.predict(X_test)


# In[42]:


print(("Test set Accuracy : ", metrics.accuracy_score(y_test, yhat_dtree)))
print(("Jaccard Similarity Score : ", jaccard_similarity_score(y_test, yhat_dtree)))
print(("F1  Accuracy : ", f1_score(y_test, yhat_dtree, average='weighted')))


# #### Logistic Regression

# In[43]:


yhat_LR = LR.predict(X_test)
LR_prob = LR.predict_proba(X_test)


# In[44]:


print(("Test set Accuracy : ", metrics.accuracy_score(y_test, yhat_LR)))
print(("Jaccard Similarity Score : ", jaccard_similarity_score(y_test, yhat_LR)))
print(("F1  Accuracy : ", f1_score(y_test, yhat_LR, average='weighted')))
print(("Log Loss : ", log_loss(y_test, LR_prob)))


# ### Results and Discussion
# The above results show that almost all the three models perform the same
# way. But while execution K Nearest Neighbour Model took a lot of time. So the
# recommendation would be to implement either Decision Tree or Logistic
# Regression.
