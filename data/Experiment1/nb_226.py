#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
names = [x for x in range(0,23)]
df = pd.read_csv("agaricus-lepiota.data", names=names)
df.head()


# In[3]:


# no missing values.
df.isna().sum().sum()


# In[4]:


#df_Y is the dataframe which includes the labels(classes) of the mushrooms.
df_Y = df[0]
df_Y


# In[5]:


# drop way for df_X.
# df_X = df.drop(0,axis = 1)
# df_X
# df[:1] slices the row!!
# get all rows in first comma and after the 1st column.
df_X = df.loc[:, 1:]
df_X


# In[6]:


# # one hot encoding in the entire data set.
# y = pd.get_dummies(df)
# y


# In[1]:


# one hot encoding using one hot encoding from sklearn and give us numpy array instantly.
from sklearn.preprocessing import OneHotEncoder
X = OneHotEncoder().fit_transform(df_X).toarray()
print(X)
# one hot encoding only one column using LabelBinarizer.
from sklearn.preprocessing import LabelBinarizer
y = LabelBinarizer().fit_transform(df_Y)
# make the column  to array.
y = y.ravel()
y


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[20]:


#x_train is the features matrix for training
#y_train is the labels for training
#x_test is test data
#y_test is test labels

#Support Vector Machine(SVM)
from sklearn import svm


svm_clf = svm.LinearSVC()
svm_clf.fit(x_train, y_train)

#K-NNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(x_train, y_train)

# #Multilayer Perceptron(MLP)
# from sklearn.neural_network import MLPClassifier

# mlp_clf = MLPClassifier(random_state=1, max_iter=300)
# mlp_clf.fit(x_train, y_train)




#Evaluation 
from sklearn.metrics import accuracy_score

y_pred = knn_clf.predict(x_test)  #model ----> svm_clf, knn_clf, mlp_clf
print((accuracy_score(y_test, y_pred)))


# In[ ]:




