#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


df=pd.read_csv('Iris.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.corr()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[21]:


df=df.drop(['Id'],axis=1)


# In[ ]:


df['Species'].value_counts()


# In[ ]:


ax=df.plot(figsize=(15,8),title='iris dataset')
ax.set_xlabel('x_axis')
ax.set_ylabel('y-axis')


# In[ ]:


corel=df.corr()
top_cor_ft=corel.index
plt.figure(figsize=(10,10))
g=sns.heatmap(df[top_cor_ft].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


df.plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm')


# In[ ]:


sns.set_style("whitegrid")
sns.pairplot(df,hue='Species',size=3)


# In[ ]:


sns.violinplot(x='Species',y='PetalLengthCm',data=df,size=5)


# In[ ]:


sns.jointplot(x='PetalLengthCm',y='PetalWidthCm',data=df,kind='kde')


# In[ ]:


sns.set_style('whitegrid');
sns.FacetGrid(df,hue='Species',size=5).map(plt.scatter,
'SepalLengthCm','SepalWidthCm').add_legend()


# In[ ]:


data=df.copy()
data.head()


# In[ ]:


data=data.drop(['Species'],axis=1)


# In[ ]:


data.iloc[0].plot(kind='bar')


# In[ ]:


sns.jointplot(x='PetalLengthCm',y='PetalWidthCm',data=df,kind='reg')


# In[ ]:


for ft in data.columns:
    sns.FacetGrid(df,hue='Species',size=7).map(sns.distplot,ft).add_legend()


# In[ ]:


df.plot.hist()


# DecisionTree Classifier:

# The classification technique is a systematic approach to build classification models from an input dat set. For example, decision tree classifiers, rule-based classifiers, neural networks, support vector machines, and naive Bayes classifiers are different technique to solve a classification problem. Each technique adopts a learning algorithm to identify a model that best fits the relationshio between the attribute set and class label of the input data. Therefore, a key objective of the learning algorithm is to build prdictive model that accurately predict the class labels of previously unkonw records.
# 
# Decision Tree Classifier is a simple and widely used classification technique. It applies a straitforward idea to solve the classification problem. Decision Tree Classifier poses a series of carefully crafted questions about the attributes of the test record.

# In[22]:


from sklearn import preprocessing  
label_encoder = preprocessing.LabelEncoder() 
df['Species']= label_encoder.fit_transform(df['Species']) 
df['Species'].unique()


# In[24]:


y=pd.Series(df.Species)
y.head()


# In[25]:


x=df.drop(['Species'],axis=1)


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
clfr=DecisionTreeClassifier(criterion='entropy',random_state=0)
clfr.fit(x_train,y_train)


# In[28]:


y_pred=clfr.predict(x_test)


# In[29]:


y_pred


# In[30]:


from sklearn.metrics import accuracy_score,classification_report
print(('Accuracy is', round(accuracy_score(y_pred, y_test)*100, 2)))
print((classification_report(y_pred, y_test)))


# In[32]:


ft_columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


# In[35]:


from sklearn import tree
plt.figure(figsize = (20,15), facecolor = 'white', dpi = 150)
clfr.fit(x, y)
tree_plt = tree.plot_tree(clfr, feature_names = ft_columns, fontsize = 13, precision = 2, filled = True, proportion = True, rounded = 10)
plt.show()


# In[33]:





# In[33]:





# In[33]:





# In[33]:





# In[33]:





# In[ ]:




