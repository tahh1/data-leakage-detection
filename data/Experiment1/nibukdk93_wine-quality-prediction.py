#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# I'll be solving only for red wine dataset. Please fork it and improve it and use them on white wine set

# # Data Overviews

# In[2]:


red_df=pd.read_csv('/kaggle/input/wine-quality-selection/winequality-red.csv')
#white_df=pd.read_csv('/kaggle/input/wine-quality-selection/winequality-white.csv')


# In[3]:


red_df.describe()


# In[4]:


red_df.info()


# In[5]:


red_df.head(2)


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt 


plt.rcParams["patch.force_edgecolor"] = True
sns.set_style('darkgrid')


# # Wine Quality -> Numerical -> Categorical

# In[7]:


wine_quality = {
    3:'Three',
    4:'Four',
    5:'Five',
    6:'Six',
    7:'Seven',
    8:'Eight'
}


# In[8]:


red_df['quality']= red_df['quality'].replace(wine_quality)


# In[9]:


red_df['quality'].unique()


# # Pairplot mainly for distribution

# In[10]:


sns.pairplot(red_df)


# In[11]:


red_df.loc[:,'citric acid':'total sulfur dioxide'].describe()


# In[12]:


plt.figure(figsize=(15,15))
sns.boxplot(data = red_df.iloc[:,2:])
plt.ylim(0,100)


# In[13]:


sns.distplot(red_df['total sulfur dioxide'], bins=50)


# Above figures and desription displays skewness of data

# In[14]:


#red_df[red_df['total sulfur dioxide']>100]['total sulfur dioxide'].count()


# # Inter Quartile Range

# In[15]:


iq1 = red_df.quantile(0.25)
iq3 = red_df.quantile(0.75)
IQR  = iq3- iq1


# In[16]:


IQR


# # Skewness

# In[17]:


red_df.skew()


# Skewness of outside of range from -1 to 1 are not good. Well atleast to wikipedia lol

# ## 10th Quartile

# In[18]:


print((red_df[['residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide','sulphates']].quantile(0.10)))


# ## 90th Quartile

# In[19]:


print((red_df[['residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide','sulphates']].quantile(0.90)))


# In[20]:


#red_df[red_df['residual sugar'] <= 3.6]['residual sugar']


# lets consider values outside of 90th quartile as outliers 

# # Replacing Outliers -> 90th quartile

# In[21]:


red_df['residual sugar'] = red_df['residual sugar'].apply(lambda x: 3.6 if (x >3.6) else x)
red_df['chlorides'] = red_df['chlorides'].apply(lambda x: 0.109 if (x >0.109) else x)
red_df['free sulfur dioxide'] = red_df['free sulfur dioxide'].apply(lambda x: 31.000 if (x >31.000) else x)
red_df['total sulfur dioxide'] = red_df['total sulfur dioxide'].apply(lambda x: 93.200 if (x >93.200) else x)
red_df['sulphates'] = red_df['sulphates'].apply(lambda x: 0.850 if (x >0.850) else x)


# In[22]:


red_df.skew()


# In[23]:


red_df.info()


# In[24]:


corr_matrix = red_df.corr()


# In[25]:


sns.heatmap(corr_matrix, cmap='magma',annot=True, lw=2, linecolor='white')


# I wont touch corrs on version 1

# In[26]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[27]:


red_df.iloc[:,:11] = scaler.fit_transform(red_df.iloc[:,:11])


# # Dependent Class Imbalances

# In[28]:


sns.countplot(red_df['quality'])


# In[29]:


red_df['quality'].value_counts()


# Dependent Feature column have higly imbalanced classes. So, lets use SMOTE for oversampling

# In[30]:


from imblearn.over_sampling import SMOTE


# In[31]:


smote = SMOTE()


# In[32]:


X = red_df.iloc[:,:11]
y= red_df['quality']


# In[33]:


X_res, y_res = smote.fit_resample(X,y)


# # Data Splitting

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X_res,y_res, test_size=0.3, random_state=101)


# # KNN

# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


cls = KNeighborsClassifier()


# In[38]:


cls.fit(X_train, y_train)


# In[39]:


y_pred = cls.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report, confusion_matrix


# # Classification Report and Confusion Matrix

# In[41]:


print(("Classification Report: \n", classification_report(y_test, y_pred)))
print(("Confusion Matrix: \n", confusion_matrix(y_test, y_pred)))


# Accuracy is 79% in average

# # Grid Search CV

# In[42]:


from sklearn.model_selection import GridSearchCV


# In[43]:


params = {
    'n_neighbors' :[3,5,7,9,11,13,15,19],
    'weights':['uniform','distance']
}


# In[44]:


grs_cv = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=10,verbose=2)


# In[45]:


grs_cv.fit(X_res,y_res)


# In[46]:


print((grs_cv.best_params_))
print((grs_cv.best_score_))


# # KNN after GSCV

# In[47]:


cls_2 = KNeighborsClassifier(n_neighbors=3,weights='distance')


# In[48]:


cls_2.fit(X_train, y_train)



# In[49]:


y_pred_2 = cls_2.predict(X_test)
print(("Classification Report: \n", classification_report(y_test, y_pred_2)))
print(("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_2)))


# Accuracy is 87% on average
