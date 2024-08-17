#!/usr/bin/env python
# coding: utf-8

# # Tokyo Airbnb Data Analysis 
# 
# This is a project of Udacity's Data scientist nanodegree
# 
# The three questions are as follows
# 1. What are the most 5 expensive and inexpensive neighbourhoods in Tokyo 
# 2. What are the types of properties available in top 5 expensive and inexpensive neighbourhoods of Tokyo 
# 3. What are the average price of types of properties in the top 5 expensive and inexpensive neighbourhood of Tokyo

# In[1]:


import numpy as np
import pandas as pd
import array
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the dataset

# In[26]:


df = pd.read_csv('dataset/may_2019/listings.csv')
df_price = df


# # Data cleaning 

# In[3]:


df_price['price']=df['price'].str.replace('$','')
df_price['price']=df['price'].str.replace(',','')
# df_listings['price']=df_listings['price'].str.replace(array(',','$'),array('')
df_price['price']=df['price'].apply(pd.to_numeric)


# # The 5 most expensive neighbourhood in Tokyo

# In[4]:


neighb_mean_price = df_price.groupby('neighbourhood').mean()['price'].sort_values(ascending=False).reset_index()
neighb_mean_price.head(5)


# # The 5 most inexpensive neighbourhood in Tokyo

# In[5]:


neighb_mean_price = df_price.groupby('neighbourhood').mean()['price'].sort_values(ascending=True).reset_index()
neighb_mean_price.head(5)


# ## Different types of property types in neighborhoods

# Lets see what are the types of properties available in Tokyo

# In[27]:


df.property_type.unique()


# ### Varities of properties in expensive neighborhoods

# In[95]:


types_of_accomdation=df_price.groupby('neighbourhood')
for group_name, df_group in types_of_accomdation:
    plt.figure()
    plt.title(group_name)
    l=types_of_accomdation.get_group(group_name)['property_type']
    ((l.value_counts())/(len(l))*100).plot(kind='bar')
    plt.show() 


# ### Varities of properties in top 5 expensive neighborhoods

# In[12]:


groups = ['Chiyoda District','Akihabara','Shibuya','Shibuya District','Nakameguro']
types_of_accomdation=df_price.groupby('neighbourhood')
#fig,ax = plt.subplot(2,3)
for i in range(0,len(groups)):
    l=types_of_accomdation.get_group(groups[i])['property_type']
    ((l.value_counts())/(len(l))*100).plot(kind='bar')
    plt.title(groups[i])
    plt.show()


# ### Varities of properties in top 5 inexpensive neighborhoods

# In[13]:


groups = ['Shiodome','Marunouchi','Nerima District','Roppongi Hills','Tsukishima']
types_of_accomdation=df_price.groupby('neighbourhood')
#fig,ax = plt.subplot(2,3)
for i in range(0,len(groups)):
    l=types_of_accomdation.get_group(groups[i])['property_type']
    ((l.value_counts())/(len(l))*100).plot(kind='bar')
    plt.title(groups[i])
    plt.show()


# ### Average prices of different properties in top 5 expensive neighborhoods

# In[17]:


grouped_multiple = df_price.groupby(['neighbourhood','property_type']).agg({'price':['mean']})
grouped_multiple.columns=['mean price']
grouped_multiple = grouped_multiple.reset_index()
t=grouped_multiple.groupby('neighbourhood')


# In[23]:


groups = ['Chiyoda District','Akihabara','Shibuya','Shibuya District','Nakameguro']
for i in range (0,len(groups)):
    a=t.get_group(groups[i])['mean price']
    b=t.get_group(groups[i])['property_type']
    df = pd.DataFrame({'property_type':b,'mean price':a})
    df.plot.barh(x='property_type', y='mean price', rot=0)
    plt.title(groups[i])
    plt.xlabel("average prices in yen")


# ### Average prices of different properties in top 5 inexpensive neighborhoods 

# In[24]:


groups = ['Shiodome','Marunouchi','Nerima District','Roppongi Hills','Tsukishima']
for i in range (0,len(groups)):
    a=t.get_group(groups[i])['mean price']
    b=t.get_group(groups[i])['property_type']
    df = pd.DataFrame({'property_type':b,'mean price':a})
    df.plot.barh(x='property_type', y='mean price', rot=0)
    plt.title(groups[i])
    plt.xlabel("average prices in yen")


# ### Average prices of different properties in all neighborhoods 

# In[97]:


for group_name, _ in t:
    a=t.get_group(group_name)['mean price']
    b=t.get_group(group_name)['property_type']
    df = pd.DataFrame({'property_type':b,'mean price':a})
    df.plot.barh(x='property_type', y='mean price', rot=0)
    plt.title(group_name)
    del(a)
    del(b)
    del(df)
    


# ### Price prediction 

# The following features have been considered to predict the price
# 1. host_id
# 2. latitude
# 3. longitude
# 4. property_type
# 5. bed_type
# 6. availability_365
# 7. minimum_nights
# 8. maximum_nights
# 9. neighbourhood
# 10. review_scores_rating

# Now lets the see the percentage of null values in each features

# In[16]:


print("% of null values in host_id features")
print((df.host_id.isnull().sum()/df.host_id.shape[0]*100))
print("% of null values in latitude features")
print((df.latitude.isnull().sum()/df.latitude.shape[0]*100))
print("% of null values in longitude features")
print((df.longitude.isnull().sum()/df.longitude.shape[0]*100))
print("% of null values in property_type features")
print((df.property_type.isnull().sum()/df.property_type.shape[0]*100))
print("% of null values in bed_type features")
print((df.bed_type.isnull().sum()/df.bed_type.shape[0]*100))
print("% of null values in availability_365 features")
print((df.availability_365.isnull().sum()/df.availability_365.shape[0]*100))
print("% of null values in minimum_nights features")
print((df.minimum_nights.isnull().sum()/df.minimum_nights.shape[0]*100))
print("% of null values in maximum_nights features")
print((df.maximum_nights.isnull().sum()/df.maximum_nights.shape[0]*100))
print("% of null values in neighbourhood features")
print((df.neighbourhood.isnull().sum()/df.neighbourhood.shape[0]*100))
print("%  of null values in review_scores_rating features")
print((df.review_scores_rating.isnull().sum()/df.review_scores_rating.shape[0]*100))


# In[68]:


clean_nulls = df[['host_id', 'latitude', 'longitude','property_type','bed_type','availability_365','neighbourhood','price']].dropna()
x = clean_nulls[['host_id', 'latitude', 'longitude','property_type','bed_type','availability_365','neighbourhood']]
y = clean_nulls['price']
y = y.str.replace('$','')
y = y.str.replace(',','')
y = y.apply(pd.to_numeric)
y


# In[21]:


print((type(y[1])))


# In[69]:


#Encoding categorical features
cat_cols = ['property_type', 'bed_type','neighbourhood']
x = x.copy()
for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            x = pd.concat([x.drop(col, axis=1), pd.get_dummies(x[col], prefix=col, prefix_sep='_')], axis=1)
        except:
            continue
x


# In[70]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# In[75]:


#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .2, random_state=39) 

lm_model = Ridge(normalize=True) # Instantiate

lm_model.fit(X_train, y_train) #Fit
        
#Predict
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)


# In[76]:


print(("The r-squared score for the model for train variables was {} on {} values.".format(r2_score(y_train, y_train_preds), len(y_train))))
print(("The r-squared score for the model for test variables was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test))))


# In[ ]:




