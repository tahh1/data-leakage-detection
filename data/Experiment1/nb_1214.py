#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import random
import itertools
import seaborn as sns

sns.set(style = 'darkgrid')
get_ipython().run_line_magic('', 'matplotlib inline')


# In[2]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[3]:


bc = pd.read_csv('data123.csv')
bc.head(1)


# ## Scaling the data to chart it and allow better predictive power

# In[4]:


bcs = pd.DataFrame(preprocessing.scale(bc.iloc[:,2:32]))
bcs.columns = list(bc.iloc[:,2:32].columns)
bcs['diagnosis'] = bc['diagnosis']


# ## Checking for correlation between variables and diagnosis

# In[5]:


from pandas.plotting import scatter_matrix
p = sns.PairGrid(bcs.iloc[:,20:32], hue = 'diagnosis', palette = 'Reds')
p.map_upper(plt.scatter, s = 20, edgecolor = 'w')
p.map_diag(plt.hist)
p.map_lower(sns.kdeplot, cmap = 'GnBu_d')
p.add_legend()

p.figsize = (30,30)


# ## Lets see how each variable breaks down by diagnosis

# In[9]:


mbc = pd.melt(bcs, "diagnosis", var_name="measurement")
fig, ax = plt.subplots(figsize=(10,5))
p = sns.violinplot(ax = ax, x="measurement", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set2');
p.set_xticklabels(rotation = 90, labels = list(bcs.columns));


# In[10]:


sns.swarmplot(x = 'diagnosis', y = 'concave points_worst',palette = 'Set2', data = bcs);


# In[11]:


sns.jointplot(x = bc['concave points_worst'], y = bc['area_mean'], stat_func=None, color="#4CB391", edgecolor = 'w', size = 6);


# ## Building a SVM to predict malignant or benign tumors

# In[17]:


X = bcs.iloc[:,0:30]

y = bcs['diagnosis']
class_names = list(y.unique())
X


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Model scores very well. very slight over-fitting.

# In[19]:


svc = SVC(kernel = 'linear',C=.1, gamma=10, probability = True)
svc.fit(X,y)
y_pred = svc.fit(X_train, y_train).predict(X_test)
t = pd.DataFrame(svc.predict_proba(X_test))
svc.score(X_train,y_train), svc.score(X_test, y_test)


# In[20]:


mtrx = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(mtrx,classes=class_names,title='Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(mtrx, classes=class_names, normalize = True, title='Normalized confusion matrix')

plt.show()


# In[ ]:




