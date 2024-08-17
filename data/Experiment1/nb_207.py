#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
cancer = datasets.load_breast_cancer()


# In[2]:


# print the names of the features
print(("Features: ", cancer.feature_names))

# print the label type of cancer
print(("Labels: ", cancer.target_names))


# In[4]:


#print data(feature)shape
cancer.data.shape


# In[5]:


# print the cancer data features (top 5 records)
print((cancer.data[0:5]))


# In[6]:


# print the cancer labels (0:malignant, 1:benign)
print((cancer.target))


# In[7]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[8]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[9]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print(("Accuracy:",metrics.accuracy_score(y_test, y_pred)))


# In[10]:


# Model Precision: what percentage of positive tuples are labeled as such?
print(("Precision:",metrics.precision_score(y_test, y_pred)))

# Model Recall: what percentage of positive tuples are labelled as such?
print(("Recall:",metrics.recall_score(y_test, y_pred)))


# In[ ]:




