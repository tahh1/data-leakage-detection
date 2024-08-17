#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
names = [x for x in range(0,7)]
df = pd.read_csv("car.data", names=names)
df.head()


# In[98]:


# no missing values.
df.isna().sum().sum()


# In[99]:


#df_Y is the dataframe which includes the labels(classes) of the mushrooms.
df_Y = df[6]
df_Y


# In[100]:


# drop way for df_X.
df_X = df.drop(6,axis = 1)
df_X


# In[101]:


df_X.dtypes


# In[102]:


# one hot encoding using one hot encoding from sklearn and give us numpy array instantly.
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
X = OneHotEncoder().fit_transform(df_X).toarray()
print(X)
# one hot encoding only one column using LabelBinarizer.
from sklearn.preprocessing import LabelEncoder
print((type(df_Y)))
y = LabelEncoder().fit_transform(df_Y)

# make the column  to array.
y = y.ravel()
y


# In[103]:


print((X.shape))
y.shape


# In[104]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[105]:


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

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train, y_train)

#Multilayer Perceptron(MLP)
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(random_state=1, max_iter=300)
mlp_clf.fit(x_train, y_train)




#Evaluation 
from sklearn.metrics import accuracy_score

y_pred = knn_clf.predict(x_test)  #model ----> svm_clf, knn_clf, mlp_clf
print(("knn {}".format(accuracy_score(y_test, y_pred))))
y_pred = mlp_clf.predict(x_test)  #model ----> svm_clf, knn_clf, mlp_clf
print(("mlp {}".format(accuracy_score(y_test, y_pred))))
y_pred = svm_clf.predict(x_test)  #model ----> svm_clf, knn_clf, mlp_clf
print(("svm {}".format(accuracy_score(y_test, y_pred))))


# In[ ]:




