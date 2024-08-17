#!/usr/bin/env python
# coding: utf-8

# ## Multinomial Naïve Bayes

# In[1]:


# importing libraries

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ##### First, we load the data:

# In[2]:


# Dataframe
path_df = "Pickles/df.pickle"
with open(path_df, 'rb') as data:
    df = pickle.load(data)

# features_train
path_features_train = "Pickles/features_train.pickle"
with open(path_features_train, 'rb') as data:
    features_train = pickle.load(data)

# labels_train
path_labels_train = "Pickles/labels_train.pickle"
with open(path_labels_train, 'rb') as data:
    labels_train = pickle.load(data)

# features_test
path_features_test = "Pickles/features_test.pickle"
with open(path_features_test, 'rb') as data:
    features_test = pickle.load(data)

# labels_test
path_labels_test = "Pickles/labels_test.pickle"
with open(path_labels_test, 'rb') as data:
    labels_test = pickle.load(data)


# ##### Let's check the dimension of our feature vectors:

# In[3]:


print((features_train.shape))
print((features_test.shape))


# ##### Cross-Validation for Hyperparameter tuning

# In[4]:


mnbc = MultinomialNB()
mnbc


# ### Model fit and performance

# In[5]:


mnbc.fit(features_train, labels_train)


# In[6]:


mnbc_pred = mnbc.predict(features_test)


# #### Training accuracy

# In[7]:


# Training accuracy
print("The training accuracy is: ")
print((accuracy_score(labels_train, mnbc.predict(features_train))))


# #### Test accuracy

# In[8]:


# Test accuracy
print("The test accuracy is: ")
print((accuracy_score(labels_test, mnbc_pred)))


# #### Classification report

# In[9]:


# Classification report
print("Classification report")
print((classification_report(labels_test,mnbc_pred)))


# #### Confusion matrix

# In[10]:


aux_df = df[['Category', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
conf_matrix = confusion_matrix(labels_test, mnbc_pred)
plt.figure(figsize=(12.8,6))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=aux_df['Category'].values, 
            yticklabels=aux_df['Category'].values,
            cmap="Blues")
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.title('Confusion matrix')
plt.show()


# In[11]:


d = {
     'Model': 'Multinomial Naïve Bayes',
     'Training Set Accuracy': accuracy_score(labels_train, mnbc.predict(features_train)),
     'Test Set Accuracy': accuracy_score(labels_test, mnbc_pred)
}

df_models_mnbc = pd.DataFrame(d, index=[0])


# In[12]:


df_models_mnbc


# ##### Let's save the model and this dataset:

# In[13]:


with open('Models/best_mnbc.pickle', 'wb') as output:
    pickle.dump(mnbc, output)
    
with open('Models/df_models_mnbc.pickle', 'wb') as output:
    pickle.dump(df_models_mnbc, output)

