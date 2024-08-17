#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum().sum()


# In[8]:


df['quality']


# In[9]:


df['quality'].unique()


# In[10]:


df.head()


# In[11]:


df['quality'] -= 2


# In[12]:


df.head()


# In[13]:


df['quality'].unique()


# In[14]:


df.columns


# In[15]:


sns.barplot(x = "quality", y = "fixed acidity", data = df, palette = "Blues")
plt.show()


# In[16]:


sns.barplot(x = "quality", y = "volatile acidity", data = df, palette = "Purples")
plt.show()


# In[17]:


sns.barplot(x = "quality", y = "citric acid", data = df, palette = "Greens")
plt.show()


# In[18]:


sns.barplot(x = "quality", y = "residual sugar", data = df, palette = "Dark2")
plt.show()


# In[19]:


sns.barplot(x = "quality", y = "chlorides", data = df, palette = "RdYlBu")
plt.show()


# In[20]:


sns.barplot(x = "quality", y = "free sulfur dioxide", data = df, palette = "PuBu")
plt.show()


# In[21]:


sns.barplot(x = "quality", y = "total sulfur dioxide", data = df, palette = "magma_r")
plt.show()


# In[22]:


sns.barplot(x = "quality", y = "density", data = df, palette = "plasma")
plt.show()


# In[23]:


sns.barplot(x = "quality", y = "pH", data = df, palette = "twilight_r")
plt.show()


# In[24]:


sns.barplot(x = "quality", y = "sulphates", data = df, palette = "brg_r")
plt.show()


# In[25]:


sns.barplot(x = "quality", y = "alcohol", data = df, palette = "winter")
plt.show()


# In[26]:


df = df.drop(["fixed acidity", "residual sugar", "free sulfur dioxide", "total sulfur dioxide", "density"], axis = 1)


# In[27]:


df.columns


# In[28]:


df.head()


# # Scaling

# In[29]:


standard_scaler = StandardScaler()


# In[30]:


df[df.columns[:-1]] = standard_scaler.fit_transform(df[df.columns[:-1]])


# In[31]:


df.head()


# # Train Test Split

# In[32]:


len(df)


# In[33]:


X = df.drop(['quality'], axis = 1)
y = df['quality']


# In[34]:


X.head()


# In[35]:


y.head()


# In[36]:


X.shape, y.shape


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[38]:


X_train.head()


# In[39]:


y_train.head()


# In[40]:


X_train.shape, y_train.shape


# In[41]:


X_test.shape, y_train.shape


# # Support Vector Machines

# In[42]:


accuracies = {}


# In[43]:


svm_rbf = SVC(kernel = "rbf")


# In[44]:


svm_rbf.fit(X_train, y_train)


# In[45]:


svm_rbf_y_pred = svm_rbf.predict(X_test)


# In[46]:


accuracy_score(y_test, svm_rbf_y_pred) * 100


# In[47]:


print(("Accuracy Score:", accuracy_score(y_test, svm_rbf_y_pred)*100))


# In[48]:


accuracies["SVM_RBF"] = accuracy_score(y_test, svm_rbf_y_pred)*100


# In[ ]:





# In[49]:


svm_linear = SVC(kernel = "linear")


# In[50]:


svm_linear.fit(X_train, y_train)


# In[51]:


svm_linear_y_pred = svm_linear.predict(X_test)


# In[52]:


acc = accuracy_score(y_test, svm_linear_y_pred) * 100


# In[53]:


print(("Accuracy Score:", acc))


# In[54]:


accuracies["SVM_LINEAR"] = acc


# In[ ]:





# In[55]:


svm_poly = SVC(kernel = "poly", degree = 3)


# In[56]:


svm_poly.fit(X_train, y_train)


# In[57]:


svm_poly_y_pred = svm_poly.predict(X_test)


# In[58]:


acc = accuracy_score(y_test, svm_poly_y_pred) * 100


# In[59]:


print(("Accuracy Score:", acc))


# In[60]:


accuracies["SVM_POLY"] = acc


# In[ ]:





# # Decision Tree Classifier

# In[61]:


tree = DecisionTreeClassifier(splitter = "best")


# In[62]:


tree.fit(X_train, y_train)


# In[63]:


tree_y_pred = tree.predict(X_test)


# In[64]:


acc = accuracy_score(y_test, tree_y_pred) * 100


# In[65]:


print(("Accuracy Score:", acc))


# In[66]:


accuracies["TREE"] = acc


# In[ ]:





# # Random Forest Classifier

# In[67]:


rfc = RandomForestClassifier(n_estimators = 100)


# In[68]:


rfc.fit(X_train, y_train)


# In[69]:


rfc_y_pred = rfc.predict(X_test)


# In[70]:


acc = accuracy_score(y_test, rfc_y_pred) * 100


# In[71]:


print(("Accuracy Score:", acc))


# In[72]:


accuracies["RANDOM_FOREST_CLASSIFIER"] = acc


# In[ ]:





# # Naive Bayes

# In[73]:


nb = GaussianNB()


# In[74]:


nb.fit(X_train, y_train)


# In[75]:


nb_y_pred = nb.predict(X_test)


# In[76]:


acc = accuracy_score(y_test, nb_y_pred) * 100


# In[77]:


print(("Accuracy Score:", acc))


# In[78]:


accuracies["NAIVE_BAYES"] = acc


# In[ ]:





# # K Nearest Neighbors Classifier

# In[79]:


knn = KNeighborsClassifier(n_neighbors = 5)


# In[80]:


knn.fit(X_train, y_train)


# In[81]:


knn_y_pred = knn.predict(X_test)


# In[82]:


acc = accuracy_score(y_test, knn_y_pred) * 100


# In[83]:


print(("Accuracy Score:", acc))


# In[84]:


accuracies["KNN"] = acc


# In[ ]:





# In[107]:


accuracies


# In[125]:


pd.Series(accuracies, index = list(accuracies.keys())).sort_values(ascending = True)


# In[ ]:




