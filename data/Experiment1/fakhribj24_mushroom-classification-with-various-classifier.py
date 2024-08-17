#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 


# In[2]:


dataset = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
dataset.head()


# In[3]:


dataset.columns


# In[4]:


dataset.info()


# from the info() function, we know that there is no missing value in this dataset

# In[5]:


encoding = LabelEncoder()
for i in dataset.columns:
    dataset[i] = encoding.fit_transform(dataset[i])


# In[6]:


dataset.head()


# In[7]:


dataset["stalk-root"].value_counts()


# In[8]:


dataset.corr()


# from the corr() function, we know that variable "veil-type" is not contributing into the other value in the dataset

# In[9]:


dataset = dataset.drop('veil-type', axis=1)


# **Feature Selection to do the Machine Learning Process**

# In[10]:


x = dataset.drop(['class'], axis=1)  #delete target column from train dataset
y = dataset['class'] # test dataset  


# In[11]:


# divide dataset into 65% train, and other 35% test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)


# knowing the class that will be classified based on the data

# In[12]:


dataset['class'].unique()


# **Make the various classification for classifying the mushroom**

# 1. KNN (K - Nearest Neighbor)

# In[13]:


classifier1 = KNeighborsClassifier(n_neighbors=2)
classifier1.fit(x_train, y_train)
#Predicting the Test set results 
y_pred = classifier1.predict(x_test)
#Making the confusion matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[14]:


sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('KNN Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[15]:


print(('accuracy of train dataset is',classifier1.score(x_train, y_train)))
print(('accuracy of test dataset is',classifier1.score(x_test, y_test)))


# 2. SVM with linear kernel

# In[16]:


classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(x_train, y_train)
#Predicting the Test set results 
y_pred = classifier2.predict(x_test)
#Making the confusion matrix 
#from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[17]:


sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('SVM with linear kernel Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[18]:


print(('accuracy of train dataset is',classifier2.score(x_train, y_train)))
print(('accuracy of test dataset is',classifier2.score(x_test, y_test)))


# 3. SVM with rbf (radial basis function) kernel 

# In[19]:


classifier3 = SVC(kernel = 'rbf', random_state = 0)
classifier3.fit(x_train, y_train)
#Predicting the Test set results 
y_pred = classifier3.predict(x_test)
#Making the confusion matrix 
#from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[20]:


sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('SVM with rbf kernel Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[21]:


print(('accuracy of train dataset is',classifier3.score(x_train, y_train)))
print(('accuracy of test dataset is',classifier3.score(x_test, y_test)))


# 4. Decision Tree Classifier with entropy impurity 

# In[22]:


classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier4.fit(x_train, y_train)
#Predicting the Test set results 
y_pred = classifier4.predict(x_test)
#Making the confusion matrix 
#from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[23]:


sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('Decision tree with entropy impurity confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[24]:


print(('accuracy of train dataset is',classifier4.score(x_train, y_train)))
print(('accuracy of test dataset is',classifier4.score(x_test, y_test)))


# 5. Random Forest Classifier with entropy impurity 

# In[25]:


classifier5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier5.fit(x_train,y_train)
#Predicting the Test set results 
y_pred = classifier5.predict(x_test)
#Making the confusion matrix 
#from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[26]:


sns.heatmap(cm, annot=True, linewidth=5, cbar=None)
plt.title('RF with with entropy impurity Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[27]:


print(('accuracy of train dataset is',classifier5.score(x_train, y_train)))
print(('accuracy of test dataset is',classifier5.score(x_test, y_test)))

