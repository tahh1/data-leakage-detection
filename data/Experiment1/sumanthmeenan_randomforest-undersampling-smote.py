#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv("../input/creditcard.csv")
data.head(10)


# In[3]:


#Data Preprocessing
data['normalized_amt'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'], axis=1)
data.head()


# In[4]:


data = data.drop(['Time'], axis=1)
data.head()


# In[5]:


#Split Data in features and labels
x = data.iloc[:, data.columns!= 'Class']
y = data.iloc[:, data.columns== 'Class']


# In[6]:


#Split data in train, val and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)


# In[7]:


y_train.values


# In[8]:


y_train.values.ravel()


# In[9]:


x_train.head()


# In[10]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train.values.ravel())


# In[11]:


y_pred = random_forest.predict(x_test)
random_forest.score(x_test, y_test)


# accuracy_Score is 99.94967405170698

# In[12]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='None',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[13]:


y_pred


# In[14]:


y_test.head()


# In[15]:


cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)


# In[16]:


plot_confusion_matrix(cnf_matrix, classes = [0,1])


# In[17]:


y_pred1 = random_forest.predict(x)
y_pred1


# In[18]:


y_pred1.round()


# In[19]:


cnf_matrix = confusion_matrix(y, y_pred1)
plot_confusion_matrix(cnf_matrix, classes = [0,1])


# In[20]:


#UnderSampling
fraudulent_ids = np.array(data[data.Class == 1].index)
NO_of_fraud_ids = len(fraudulent_ids)
print(('No. of fradulent transactions in given data are:', NO_of_fraud_ids))

normal_ids = np.array(data[data.Class != 1].index)

random_normal_ids = np.random.choice(normal_ids, NO_of_fraud_ids, replace = False)
print(('No. of normal transactions in given data are:', len(random_normal_ids)))

under_sample_indices = np.concatenate([fraudulent_ids,random_normal_ids])
print(('No. of indices in undersampled data are:', len(under_sample_indices)))


# In[21]:


under_sampled_data = data.iloc[under_sample_indices, :]

x_undersample = under_sampled_data.iloc[:, under_sampled_data.columns != 'Class']
y_undersample = under_sampled_data.iloc[:, under_sampled_data.columns == 'Class']


# In[22]:


under_sampled_data.head()


# In[23]:


x_undersample.head()


# In[24]:


y_undersample.head()


# In[25]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(x_undersample, y_undersample, test_size = 0.3, random_state=0)


# In[26]:


random_forest1 = RandomForestClassifier(n_estimators=100)
random_forest1.fit(x_train1, y_train1.values.ravel())


# In[27]:


y_pred1 = random_forest.predict(x_test1)
random_forest1.score(x_test1, y_test1)


# In[28]:


cnf_matrix1 = confusion_matrix(y_test1, y_pred1)
print(cnf_matrix1)


# In[29]:


plot_confusion_matrix(cnf_matrix1, classes = [0,1])


# In[30]:


y_pred2 = random_forest1.predict(x)

cnf_matrix3 = confusion_matrix(y, y_pred2)
plot_confusion_matrix(cnf_matrix3, classes = [0,1])


# In[31]:


#Oversampling using SMOTE
x_resample, y_resample = SMOTE().fit_sample(x, y.values.ravel())


# In[32]:


x_resample = pd.DataFrame(x_resample)
y_resample = pd.DataFrame(y_resample)


# In[33]:


x_resample.head()


# In[34]:


y_resample.head()


# In[35]:


x_train5, x_test5, y_train5, y_test5 = train_test_split(x_resample, y_resample, test_size = 0.3, random_state=2)

random_forest2 = RandomForestClassifier(n_estimators=100)
random_forest2.fit(x_train5, y_train5.values.ravel())


# In[36]:


y_pred5 = random_forest2.predict(x_test5)
random_forest2.score(x_test5, y_test5)


# In[37]:


cnf_matrix5 = confusion_matrix(y_test5, y_pred5)
plot_confusion_matrix(cnf_matrix5, classes = [0,1])


# In[38]:


y_pred6 = random_forest2.predict(x_resample)

cnf_matrix6 = confusion_matrix(y_resample, y_pred6)
plot_confusion_matrix(cnf_matrix6, classes = [0,1])


# In[ ]:




