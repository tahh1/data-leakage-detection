#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Devanand072001/MachineLearningTask/blob/master/191CS151(weather_prediction).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# **Importing libriries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Data visualisation**

# In[ ]:


df = pd.read_csv('weatherAUS.csv')
df  


# In[ ]:


df.describe()


# In[ ]:


df = df.drop(["Evaporation","Sunshine","Cloud9am","Cloud3pm","Location", "Date"], axis =1)
df.describe()


# **Visualising data using scatter plot**

# In[ ]:


plt.figure(figsize = (8,8))
sns.scatterplot(x = 'Humidity9am', y = 'Temp9am', hue = 'RainTomorrow' , palette = 'inferno',data = df)


# In[ ]:


df = df.dropna(axis = 0)
df.columns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])
df['RainToday'] = le.fit_transform(df['RainToday'])
df['RainTomorrow'] = le.fit_transform(df['RainTomorrow'])


# In[ ]:


x = df.drop(['RainTomorrow'], axis = 1)
y = df['RainTomorrow']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# **Implementing decision tree algorithmn**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
print(x_train)
dt.fit(x_train,y_train)
predictions = dt.predict(x_test)
print((confusion_matrix(y_test, predictions)))
print((classification_report(y_test, predictions)))
print((accuracy_score(y_test, predictions)))


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
from sklearn import tree
reg = tree.DecisionTreeRegressor()
reg = reg.fit(x, y)


# In[ ]:


from IPython.display import Image
import pydotplus
dot_data = tree.export_graphviz(reg, out_file=None, rounded=True, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from IPython.display import SVG
from graphviz import Source
from IPython.display import display


graph = Source(tree.export_graphviz(dt, out_file=None,filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:




