#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("../input/automobileEDA.csv")
df.head()


# df.head() will give us the details fo top 5 rows of every cooloumn. We can use df.tail() to get the last 5 rows and similiary df.head(10) to get top 10 rows.
# 
# The data is about cars and we need to predict the price of car using the above data
# 
# We will be using Decision Tree to get the price of the car. 

# In[3]:


df.dtypes


# dtypes gives the data type of coloumn

# In[4]:


df.describe()


# In the above dataframe all the coloumns are not numeric. So we will consider only those coloumn whose values are in numeric and will  make all numeric to float.

# In[5]:


df.dtypes
for x in df:
    if df[x].dtypes == "int64":
        df[x] = df[x].astype(float)
        print((df[x].dtypes))

 


# Preparing the Data
# As with the classification task, in this section we will divide our data into attributes and labels and consequently into training and test sets. We will create 2 data set,one for price while the other (df-price).
# Since pur dataframe has many data in object format, for this analysis we are removing all the coloumn with object type and for all NaN value we are removig that row

# In[6]:


df = df.select_dtypes(exclude=['object'])
df=df.fillna(df.mean())
X = df.drop('price',axis=1)
y = df['price']


# Here the X variable contains all the columns from the dataset, except 'Price' column, which is the label. The y variable contains values from the 'Price' column, which means that the X variable contains the attribute set and y variable contains the corresponding labels.

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# To train the tree, we will  use the DecisionTreeRegressor class and call the fit method:

# In[8]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)


# Lets predict the price

# In[9]:


y_pred = regressor.predict(X_test)


# Let's check the difference between Actual and predicted values

# In[10]:


df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# In[11]:


from sklearn import metrics
print(('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)))
print(('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred)))
print(('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))))


# The mean absolute error for our algorithm is 2397.688524590164, which is less than 20 percent of the mean of all the values in the 'Price' column. This means that our algorithm did a  prediction, but it needs lot of improvement.

# In[12]:


import seaborn as sns
plt.figure(figsize=(5, 7))


ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# The above is the graph between the actual and predicted values

# In[ ]:




