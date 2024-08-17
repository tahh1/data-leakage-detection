#!/usr/bin/env python
# coding: utf-8

# ### first of all we need to import all important library 

# In[51]:


import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')


# ### then step -2 we need to load the dataset in dataframe

# In[11]:


df=pd.read_csv("Iris.csv")
df.head()


# In[12]:


# now check the shape of data set
df.shape


# In[13]:


df.info()


# # Now time to Eda

# In[8]:


df.isna().sum()


# #### the result shows that there in no null value so we do not need to visulize it through heat map

# ### now define x and y variable for dependent and independent variable

# In[15]:


X = df[['sepal_length','sepal_width', 'petal_length', 'petal_width']]
y = df['class']


# In[17]:


X.head()


# In[31]:


X.hist(edgecolor='red',linewidth=1.2, figsize=(10, 8))


# In[37]:


df.boxplot(figsize=(7,5),grid=True)


# In[39]:


df.boxplot(by="class",figsize=(12,6));


# ### now its time to instantiate the classifier

# In[64]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[65]:


gnb = GaussianNB()


# In[66]:


gnb.fit(X_train,y_train)


# In[67]:


y_pred=gnb.predict(X_test)
y_pred


# In[68]:


new_df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
new_df


# #### now in the result we see that most of cases y actual and y predicated is same it shows that how our model is accaurate

# ## now its time to evaluating our model in different parameter

# In[69]:


con_matrix=confusion_matrix(y_test,y_pred)
con_matrix


# now compare actual value and responce value to find the accuracy of our model

# In[70]:


accuracy=accuracy_score(y_test, y_pred)* 100
accuracy


# In[89]:


recall_score(y_test,y_pred,average='micro')


# In[92]:


from sklearn.metrics import classification_report
print(('Report : ', classification_report(y_test, y_pred)))


# ## its all about naive bayes classifier  

# # now lest start evaluating through  logistic regression 

# In[94]:


df.info()


# In[101]:


sns.set_style('whitegrid')
sns.countplot(x='class',data=df,palette='RdBu_r')


# In[105]:


df['sepal_length'].hist(bins=30,color='darkred')


# In[135]:


X = df.iloc[:, :-1]
y = df.iloc[:, 4]


# In[136]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[138]:


from sklearn.linear_model import LogisticRegression
lo_reg=LogisticRegression()

lo_reg.fit(X_train,y_train)



# In[139]:


y_pred=lo_reg.predict(X_test)
y_pred


# In[140]:


ne_df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
ne_df


# In[141]:


confusion_matrix(y_test,y_pred)


# In[142]:


accuracy_score(y_test,y_pred)


# In[ ]:




