#!/usr/bin/env python
# coding: utf-8

# # Lecture 3: Measures of Model Performance

# ## 1. F1 score calculation in Python

# Sklearn: 
# 
# sklearn.metrics.f1_score(y_true, y_pred)
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
# 
# Provide the labels of all samples, and their predictions to this function, and F1 score can be calculated.

# Example:

# In[2]:


from sklearn.metrics import f1_score
y_true = [0, 1, 2, 2, 0, 1, 2, 2, 1]
y_pred = [0, 2, 2, 1, 0, 0, 1, 2, 2]
f1_score(y_true, y_pred, average=None)


# Output: 0.8 for class 0, 0 for class 1, and 0.5 for class 2.

# ## 2. AUCROC in Python

# sklearn: 
# 
# sklearn.metrics.roc_auc_score(y_ture, y_score)
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
# 
# Give labels and the outputs of your model to this function, and sklearn can calculate the AUCROC score (the area below ROC curve).

# Example (Practice 3):

# In[3]:


from sklearn.metrics import roc_auc_score
y_true = [1, 0, 0, 1, 1, 1, 0, 0, 1, 0]
y_scores = [0.13, 0.08, 0.04, 0.45, 0.75, 0.64, 0.23, 0.18, 0.32, 0.55]
roc_auc_score(y_true, y_scores)


# ## 3. K-Nearest Neighbors Classifier

# sklearn.neighbors.KNeighborsClassifier

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# In[3]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
X = [[0], [1], [2], [3], [7], [9]]
y = [0, 0, 1, 1, 2, 2]
knn.fit(X, y)


# In[4]:


knn.predict([[0.7], [1.9], [6.2]])


# Prediction results for the three samples: class 0 for 0.7, class 1 for 1.9, and class 2 for 6.2.

# In[5]:


knn.predict_proba([[0.7], [1.9], [6.2]])


# The probability of each class for each given input.

# In[ ]:




