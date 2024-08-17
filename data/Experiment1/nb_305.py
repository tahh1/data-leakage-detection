#!/usr/bin/env python
# coding: utf-8

# ## Just some basic EDA on the data from ratingraph, not including the genres yet. (this first part was before i found the outliers for number of episodes per season.)

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Load data
tv_show_df = pd.read_pickle('clean_tv_show_df.pkl')

# Take a look at the datatypes
tv_show_df.info()


# In[4]:


# Load a copy in case i mess up
copy_tv_show_df = pd.read_pickle('clean_tv_show_df.pkl')


# In[5]:


tv_show_df.head()


# In[6]:


tv_show_df.shape


# In[7]:


tv_show_df.corr()


# In[8]:


# try a heatmap for the corr matrix
sns.heatmap(tv_show_df.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)

plt.gca().set_ylim(len(tv_show_df.corr())+0.5, -0.5);  # quick fix to make sure viz isn't cut off


# In[9]:


# Plot all of the variable-to-variable relations as scatterplots
sns.pairplot(tv_show_df, height=1.2, aspect=1.25);


# In[10]:


tv_show_df.describe()


# ## Start with just the basic linear regression model

# In[27]:


# Create an empty model
lr = LinearRegression()

# Choose just the X1 column for our data
X = tv_show_df[['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating']]
#X = tv_show_df['Season_1_Rating'].values.reshape(-1,1)

# Choose the response variable
y = tv_show_df['Num_of_Seasons']

# Fit the model 
lr.fit(X, y)


# In[28]:


lr.score(X, y) #R^2 = .30... pretty bad


# In[30]:


# print out intercept
print((lr.intercept_))

# print out other coefficients
print((lr.coef_))


# ## Try to model it using statsmodels

# In[32]:


#add a constant since statsmodels.api does not add one by default
# Add a column of ones with sm.add_constant()
sm.add_constant(X).head()


# In[33]:


#Create the model
model = sm.OLS(y, sm.add_constant(X)) 

#Fit
fit = model.fit()

#Print out summary
fit.summary()


# In[34]:


# Use statsmodels to plot the residuals vs the fitted values
plt.figure(figsize=(10, 7))
plt.scatter(fit.predict(), fit.resid)    #change this if working with sklearn

plt.axhline(0, linestyle='--', color='gray')
plt.xlabel('Predicted Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18);


# ## I found outliers for num of episodes per season so I'm going to try to fix that and then run this again

# In[39]:


# Load data
tv_show_df2 = pd.read_pickle('cleaned_outliers_tv_show_df.pkl')


# In[42]:


tv_show_df2.head()


# In[43]:


tv_show_df2.corr()


# In[45]:


# try a heatmap for the corr matrix
sns.heatmap(tv_show_df2.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)

plt.gca().set_ylim(len(tv_show_df2.corr())+0.5, -0.5);  # quick fix to make sure viz isn't cut off


# In[46]:


# Plot all of the variable-to-variable relations as scatterplots
sns.pairplot(tv_show_df2, height=1.2, aspect=1.25);

#episodes per season looks way better now!


# In[50]:


# Create an empty model
lr2 = LinearRegression()

# Choose just the X1 column for our data
X = tv_show_df2[['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating']]
#X = tv_show_df['Season_1_Rating'].values.reshape(-1,1)

# Choose the response variable
y = tv_show_df2['Num_of_Seasons']

# Fit the model 
lr2.fit(X, y)


# In[51]:


lr2.score(X, y) #wow the score jumped by .04! neat!


# ## Next notebook, check when we add all the dummy variables for genre

# In[ ]:




