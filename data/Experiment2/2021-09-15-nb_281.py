#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns


# In[2]:


data= pd.read_csv('C:/Users/me/Desktop/pyt/kaggle/Glass/glass.csv')


# ## EXPLORING DATA

# In[3]:


data.head()


# In[4]:


print(('There are ',data.shape[0] ,'rows and ',data.shape[1],' columns.'))


# In[5]:


data.info()


# In[6]:


data.describe().T.round(2)


# ### Getting missing data

# In[7]:


def get_missing(data):
    length = data.shape[0]
    null_count= data.isnull().sum()
    nan_count= ((data=='nan') | (data== 'NaN')).sum()
    empty_count= ((data== '') | (data==' ')).sum()
    null_percent= null_count/ length
    nan_percent= nan_count/length
    empty_percent= empty_count/ length
    abc= pd.DataFrame({'null_count' : null_count,
        'null_percent' : null_percent,
        'nan_count' :nan_count,              
        'nan_percent' : nan_percent,
        'empty_count' : empty_count,
        'empty_percent' : empty_percent               
    })
    return abc
get_missing(data)


# No missing data!
# 

# In[8]:


data.Type.value_counts()


# ## Visuaisation of the Target Value 

# In[9]:


plt.bar(data.Type.unique(),data.Type.value_counts())
plt.ylim(0,80)


# ### Looking at the correlation of features

# In[10]:


data2= data.drop('Type',axis=1)
plt.figure(figsize=(15,10))
sns.heatmap(data2.corr().abs(), annot= True)
plt.show()


# In[11]:


y= data['Type']
x= data.drop(['Type', 'Fe', 'Ba'], axis=1)


# In[12]:


from sklearn.model_selection import train_test_split, cross_val_score
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# ### Importing the CLASSIFIERS and other Libraries

# In[13]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[14]:


from sklearn.model_selection import GridSearchCV,KFold


# ### Making a function to test different classifiers simultaneously using Dictionary.

# In[15]:


def result_using_tree_classifier(x_train,x_test,y_train,y_test):
    kf = KFold(n_splits=5,random_state=None)
    scores= []
    algos2={'decision tree': {'model' : DecisionTreeClassifier(random_state=42),
                      'param' : {'criterion':['gini', 'entropy'],'max_depth' : np.arange(1,6,1).tolist()  }
                     },
           'random forest': {'model' : RandomForestClassifier(random_state=42, n_jobs= -1),
                      'param' : {'n_estimators': [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]}
                     },
           'extra tree' : {'model' : ExtraTreesClassifier(random_state=42, n_jobs= -1),
                    'param' : {'n_estimators': [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]}
                     } 
           } 
    for algo_name, params in list(algos2.items()) :
        Grid2 = GridSearchCV( params['model'], params['param'], cv=kf, return_train_score=False)
        Grid2.fit(x_train, y_train)
        ypred= Grid2.predict(x_test)
        scores.append({
            'model' : algo_name,
            'best_score': Grid2.best_score_,
            'best_para': Grid2.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_para']).set_index('model')
result_using_tree_classifier(x_train,x_test,y_train,y_test)


# ## Conclusion
# 
# Here, we can see 'random forest' performs decently but, 'extra tree' is slightly better than it.
