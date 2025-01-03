#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
iris = load_iris()
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Targets']
model = KMeans(n_clusters=3)
model.fit(X)
model.labels_
plt.figure(figsize=(18,12))
colormap = np.array(['red', 'lime', 'blue'])
plt.subplot(3, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
plt.subplot(3, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
print (predY)
plt.subplot(3, 2, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')
plt.subplot(3, 2, 4)
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[predY], s=40)
plt.title('K Mean Classification')
print(('The accuracy score of K-Mean: ',sm.accuracy_score(y, model.labels_)))
print(('The Confusion matrixof K-Mean: ',sm.confusion_matrix(y, model.labels_)))
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
plt.subplot(3, 2, 5)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')
print(('The accuracy score of EM: ',sm.accuracy_score(y, y_cluster_gmm)))
print(('The Confusion matrix of EM: ',sm.confusion_matrix(y, y_cluster_gmm)))


# In[ ]:





# In[ ]:




