#!/usr/bin/env python
# coding: utf-8

# **Sinchana S R**

# # Prediction using Decision Tree Algorithm
# 
# **Task**
# 
# * For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically.
# * The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# # Table of Contents
# 
# * [Importing the required libraries](#im)
# * [Loading the iris dataset](#ld)
# * [Data Visualization](#dv)
# * [Define the Decision Tree Algorithm](#dt)
# * [Visualize Decision Tree](#vdt)

# In[1]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))


# # <a id="im"> Importing the required libraries </a>

# In[2]:


import sklearn.datasets as datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # <a id="ld"> Loading the iris dataset </a>

# In[3]:


iris=datasets.load_iris()


# # <a id="dv"> Data Visualization </a>

# In[4]:


iris


# In[5]:


type(iris)


# In[6]:


list(iris.keys())


# In[7]:


iris['data']


# In[8]:


iris['target']


# In[9]:


iris['frame']


# In[10]:


iris['target_names']


# In[11]:


iris['DESCR']


# In[12]:


iris['feature_names']


# In[13]:


iris['filename']


# # Constructing the iris dataframe

# In[14]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[15]:


df.head()


# In[16]:


df.tail()


# In[17]:


df.shape


# In[18]:


df.size


# In[19]:


# Check the column names
df.columns


# In[20]:


#Check null values
df.isnull().sum()


# In[21]:


df.describe()


# In[22]:


y=iris.target


# In[23]:


y


# In[24]:


# Plot histogram of the given data 
df.hist(figsize = (12,12))


# In[25]:


# Pairplot of the given data
sns.pairplot(df)


# # <a id="dt"> Define the Decision Tree Algorithm </a>

# In[26]:


# import
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report


# In[27]:


from sklearn .model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(df,y,test_size = 0.30,random_state = 42)


# In[28]:


# Defining the decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)


# In[29]:


# Accuracy For training data
predict = dtree.predict(x_train)
print(("Accuracy of training data : ",accuracy_score(predict,y_train)*100,"%"))
print(("Confusion matrix of training data :'\n' ",confusion_matrix(predict,y_train)))
sns.heatmap(confusion_matrix(predict,y_train),annot = True,cmap = 'BuGn')


# In[30]:


# Accuracy For testing data
predict = dtree.predict(x_test)
print(("Accuracy of testing data : ",accuracy_score(predict,y_test)*100,"%"))
print(("Classification Report : ",classification_report(predict,y_test)))
print(("Confusin matrix of testing data :\n ",confusion_matrix(predict,y_test)))
sns.heatmap(confusion_matrix(predict,y_test),annot = True,cmap = 'BuGn')


# # <a id="vdt"> Visualize Decision Tree </a>
# 
# **Let us visualize the Decision Tree to understand it better.**

# In[31]:


from sklearn import tree
plt.figure(figsize  = (19,19))
tree.plot_tree(dtree,filled = True,rounded = True,proportion = True,node_ids = True , feature_names = iris.feature_names)
plt.show()


# You can now feed any new/test data to this classifer and it would be able to predict the right class accordingly.

# In[ ]:




