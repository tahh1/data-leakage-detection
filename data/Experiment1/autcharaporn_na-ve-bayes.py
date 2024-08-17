#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

#X = dataset.iloc[:, 0:-1]
#y = dataset.iloc[:, -1]


# In[3]:


print(X)


# In[4]:


#	Splitting	the	dataset	into	the	Training	set	and	Test	set
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[5]:


#	Feature	Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[6]:


#	Fitting	Naive	Bayes	to	the	Training	Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[7]:


#	Predicting	the	Test	Set	results
y_pred = classifier.predict(X_test)


# In[8]:


#	Making	the	Confusion	Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #บ่งบอกว่า testดีหรือไม่ดี true pos,true neg. ควรสูง , False pos.,False neg ควรต่ำ
print(cm)


# In[9]:


dataset.head()


# In[10]:


#	Visualising the	Training	set	results
from  matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.01, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.01, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('YELLOW', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1], 
                c = ListedColormap(('red','blue'))(i), label = j)
plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[ ]:




