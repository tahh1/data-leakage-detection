#!/usr/bin/env python
# coding: utf-8

# # Iris Dataset

# Problem statement: For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ## Required Libraries

# In[26]:


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn import tree
import sklearn.datasets as datasets


# ## Reading the Dataset

# In[17]:


# Importing the dataset
dataset = pd.read_csv('F:\Iris.csv')
dataset.head()


# In[18]:


# deleting the iD columns
del dataset['Id']


# In[20]:


dataset.describe()


# In[22]:


dataset.info()


# In[24]:


#No. of rows and columns 
dataset.shape


# In[25]:


dataset.Species.value_counts()


# ## Data Visualization

# In[27]:


plt.figure(figsize=(20,7))
sns.violinplot(dataset.Species,dataset.PetalLengthCm)


# In[29]:


# Pairplot of the given data
sns.pairplot(dataset)


# ## Training the Dataset

# In[45]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)


# ## Building the model

# In[46]:


# Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 7)
classifier.fit(X_train, y_train)


# In[47]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[48]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# ## ACCURACY

# In[49]:


acc=accuracy_score(y_test,y_pred)
print((" The model accuracy score :",acc*100))


# ## Visualization Of Tree

# In[51]:


text_representation = tree.export_text(classifier)
fig = plt.figure(figsize=(30,25))
_ = tree.plot_tree(classifier, feature_names=dataset.columns.drop("Species"),class_names=dataset.Species.unique(), 
                   filled=True)


# In[ ]:




