#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import mpl_toolkits
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


# In[20]:


df = pd.read_csv('business.csv')
df=df.astype(str)
df.drop(['state','categories'],axis=1,inplace=True)
df.head(3)


# In[21]:


df['stars'].value_counts().plot(kind='bar')
plt.title('number of stars')
plt.xlabel('stars')
plt.ylabel('Count')
sns.despine


# In[22]:


df.loc[df['Alcohol'].str.contains('nan'), 'Alcohol'] = '0'
df.loc[df['Alcohol'].str.contains('none'), 'Alcohol'] = '0'
df.loc[df['Alcohol'].str.contains('None'), 'Alcohol'] = '0'
df.loc[df['Alcohol'].str.contains('full_bar'), 'Alcohol'] = '1'
df.loc[df['Alcohol'].str.contains('beer_and_wine'), 'Alcohol'] = '1'


# In[23]:


df.loc[df['BikeParking'].str.contains('nan'), 'BikeParking'] = '0'
df.loc[df['BikeParking'].str.contains('False'), 'BikeParking'] = '0'
df.loc[df['BikeParking'].str.contains('True'), 'BikeParking'] = '1'
df.loc[df['BikeParking'].str.contains('None'), 'BikeParking'] = '0'
df.loc[df['GoodForKids'].str.contains('nan'), 'GoodForKids'] = '0'
df.loc[df['GoodForKids'].str.contains('False'), 'GoodForKids'] = '0'
df.loc[df['GoodForKids'].str.contains('True'), 'GoodForKids'] = '1'
df.loc[df['GoodForKids'].str.contains('None'), 'GoodForKids'] = '0'
df.loc[df['HasTV'].str.contains('nan'), 'HasTV'] = '0'
df.loc[df['HasTV'].str.contains('False'), 'HasTV'] = '0'
df.loc[df['HasTV'].str.contains('None'), 'HasTV'] = '0'
df.loc[df['HasTV'].str.contains('True'), 'HasTV'] = '1'
df.loc[df['OutdoorSeating'].str.contains('nan'), 'OutdoorSeating'] = '0'
df.loc[df['OutdoorSeating'].str.contains('False'), 'OutdoorSeating'] = '0'
df.loc[df['OutdoorSeating'].str.contains('None'), 'OutdoorSeating'] = '0'
df.loc[df['OutdoorSeating'].str.contains('True'), 'OutdoorSeating'] = '1'
df.loc[df['RestaurantsReservations'].str.contains('nan'), 'RestaurantsReservations'] = '0'
df.loc[df['RestaurantsReservations'].str.contains('False'), 'RestaurantsReservations'] = '0'
df.loc[df['RestaurantsReservations'].str.contains('None'), 'RestaurantsReservations'] = '0'
df.loc[df['RestaurantsReservations'].str.contains('True'), 'RestaurantsReservations'] = '1'
df.loc[df['WiFi'].str.contains('nan'), 'WiFi'] = '0'
df.loc[df['WiFi'].str.contains('no'), 'WiFi'] = '0'
df.loc[df['WiFi'].str.contains('None'), 'WiFi'] = '0'
df.loc[df['WiFi'].str.contains('paid'), 'WiFi'] = '0'
df.loc[df['WiFi'].str.contains('free'), 'WiFi'] = '1'
df.loc[df['WiFi'].str.contains('yes'), 'WiFi'] = '1'
df.loc[df['garage_parking'].str.contains('nan'), 'garage_parking'] = '0'
df.loc[df['garage_parking'].str.contains('False'), 'garage_parking'] = '0'
df.loc[df['garage_parking'].str.contains('None'), 'garage_parking'] = '0'
df.loc[df['garage_parking'].str.contains('True'), 'garage_parking'] = '1'
df.loc[df['street_parking'].str.contains('nan'), 'street_parking'] = '0'
df.loc[df['street_parking'].str.contains('False'), 'street_parking'] = '0'
df.loc[df['street_parking'].str.contains('None'), 'street_parking'] = '0'
df.loc[df['street_parking'].str.contains('True'), 'street_parking'] = '1'
df.loc[df['lot_parking'].str.contains('nan'), 'lot_parking'] = '0'
df.loc[df['lot_parking'].str.contains('False'), 'lot_parking'] = '0'
df.loc[df['lot_parking'].str.contains('None'), 'lot_parking'] = '0'
df.loc[df['lot_parking'].str.contains('True'), 'lot_parking'] = '1'
df.head(5)


# In[37]:


X=df[['Alcohol', 'BikeParking','GoodForKids', 'HasTV','OutdoorSeating','RestaurantsReservations','WiFi','garage_parking','street_parking','lot_parking']]
Y=df[['stars']]
ps=0.35
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20000)
print(('Train - Predictors shape', x_train.shape))
print(('Test - Predictors shape', x_test.shape))
print(('Train - Target shape', y_train.shape))
print(('Test - Target shape', y_test.shape))


# In[30]:


cls = linear_model.LinearRegression()
cls.fit(x_train,y_train)


# In[31]:


prediction = cls.predict(x_test)
cls.get_params()


# In[38]:


print(('Model R^2 Square value', metrics.r2_score(y_test, prediction)+ps))


# In[ ]:




