#!/usr/bin/env python
# coding: utf-8

# In[250]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[251]:


dataset=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
dataset.head()


# In[252]:


dataset.shape


# In[253]:


dataset.isnull().sum()


# In[254]:


sns.heatmap(dataset.isnull(),cmap='viridis')


# ### Numerical Features

# In[255]:


numerical_features=[feature for feature in dataset.columns if dataset[feature].dtypes!='O']
numerical_features


# In[256]:


dataset.age_approx.isnull().sum()


# In[257]:


dataset['age_approx'].describe()


# In[258]:


max=60
dataset['age_approx']=dataset['age_approx'].fillna(max)


# In[259]:


dataset['age_approx'].isnull().sum()


# ### Categorical Features

# In[260]:


categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
categorical_features


# In[261]:


dataset[categorical_features].head()


# In[262]:


dataset[categorical_features].isnull().sum()


# In[263]:


dataset[categorical_features].sex.describe()


# In[264]:


dataset['sex']=dataset['sex'].fillna('male')


# In[265]:


dataset[categorical_features].isnull().sum()


# In[266]:


dataset['anatom_site_general_challenge'].describe()


# In[267]:


dataset['anatom_site_general_challenge']=dataset['anatom_site_general_challenge'].fillna('torso')


# In[268]:


dataset[categorical_features].columns


# In[269]:


for feature in dataset[categorical_features]:
    print()
    print((dataset[feature].unique()))
    print()
    print((len(dataset[feature].unique())))


# In[270]:


dataset.drop(['diagnosis','benign_malignant'],axis=1,inplace=True)


# In[271]:


dataset.head()


# In[272]:


len(dataset['patient_id'].unique())


# In[273]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in ['image_name', 'patient_id', 'sex', 'anatom_site_general_challenge']:
    dataset[feature]=le.fit_transform(dataset[feature])


# In[274]:


dataset.head()


# In[275]:


test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')


# In[276]:


test.head()


# In[277]:


len(test['patient_id'].unique())


# In[278]:


test.shape


# In[279]:


test.isnull().sum()


# In[280]:


test['anatom_site_general_challenge'].describe()


# In[281]:


test['anatom_site_general_challenge']=test['anatom_site_general_challenge'].fillna('torso')


# In[282]:


test.isnull().sum()


# In[283]:


dataset.head()


# In[284]:


X=dataset.drop(['target'],axis=1)


# In[285]:


y=dataset['target']
y.head()


# In[286]:


y=pd.DataFrame(data=y)
y.head()


# In[287]:


test.head()


# In[288]:


test.columns


# In[289]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in ['image_name', 'patient_id', 'sex','anatom_site_general_challenge']:
    test[feature]=le.fit_transform(test[feature])


# In[290]:


test.head()


# In[291]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y,random_state=0)


# In[292]:


y_train


# In[293]:


from sklearn.neighbors import KNeighborsClassifier
KNC=KNeighborsClassifier()
KNC.fit(X_train,y_train)


# In[294]:


test_pred=KNC.predict(test)


# In[295]:


sub=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub.loc[:,'target']=test_pred
sub.to_csv('submission.csv',index=False)


# In[ ]:




