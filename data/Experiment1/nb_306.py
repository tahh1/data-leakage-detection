#!/usr/bin/env python
# coding: utf-8

# # EDA: This df has all genres as dummy variables, MPAA rating as ordinal data, and a binary column of whether or not a show is produced by a large company.
# 
# #### Note: The scraped data for production company only yeilded about 502 usable rows. After the inner merge, there are only 477 rows of data in this data set.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load data
df = pd.read_pickle('all_scraped_features_df.pkl')

# Take a look at the datatypes
df.info()


# In[3]:


# Load a copy in case i mess up
copy_df = pd.read_pickle('all_scraped_features_df.pkl')


# In[4]:


df.head(3)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


# try a heatmap for the corr matrix
plt.figure(figsize = (20,10))
sns.heatmap(df.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)

plt.gca().set_ylim(len(df.corr())+0.5, -0.5);  # quick fix to make sure viz isn't cut off


# In[9]:


#These are the only columns we care about for our model
features = ['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating',
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
            'History', 'Horror', 'Music', 'Mystery', 'Other', 'Romance', 
            'Sci-fi', 'Sport', 'Thriller', 'War', 'Ordinal_MPAA', 'Large_prod_co']

target = 'Num_of_Seasons'


# In[10]:


# Create an empty model
lr = LinearRegression()

# Choose just the X1 column for our data
X = df[features]
#X = tv_show_df['Season_1_Rating'].values.reshape(-1,1)

# Choose the response variable
y = df[target]

# Fit the model 
lr.fit(X, y)


# In[11]:


lr.score(X, y)


# In[12]:


# print out intercept
print((lr.intercept_))

# print out other coefficients
print((lr.coef_))


# In[13]:


#add a constant since statsmodels.api does not add one by default
#Note: statsmodels.api does not include constant by default
# Add a column of ones with sm.add_constant()
sm.add_constant(X).head()


# In[14]:


#Create the model
model = sm.OLS(y, sm.add_constant(X)) 

#Fit
fit = model.fit()

#Print out summary
fit.summary()


# In[15]:


# Use statsmodels to plot the residuals vs the fitted values
plt.figure(figsize=(10, 7))
plt.scatter(fit.predict(), fit.resid)    #change this if working with sklearn

plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('Predicted Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18);


# In[ ]:




