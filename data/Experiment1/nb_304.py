#!/usr/bin/env python
# coding: utf-8

# ## I will be comparing the models on two datasets; one that has near 1000 rows and has dummy variables for genre, and MPAA rating as ordinal data.
# 
# ## The other has the same variables, with the added binary variable of being whether or not the show was produced by a large production company or not. This data set only has around 500 rows.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(context='notebook', style='whitegrid', font_scale=1.2)


# In[2]:


#1000 row dataset
df_og = pd.read_pickle('Ordinal_MPAA_merged_with_dummy_genres.pkl')

#500 row dataset
df_prod = pd.read_pickle('all_scraped_features_df.pkl')


# In[3]:


df_og.info()


# In[4]:


df_prod.info()


# In[7]:


#filter the datasets to just the variables we care about, keeping target at the end
features1 = ['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating',
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
            'History', 'Horror', 'Music', 'Mystery', 'Other', 'Romance', 
            'Sci-fi', 'Sport', 'Thriller', 'War', 'Ordinal_MPAA']

features2 = features1.copy()
features2.append('Large_prod_co')

target = 'Num_of_Seasons'


# ## First dataset

# In[9]:


g = sns.PairGrid(df_og[features1].sample(frac=0.6), diag_sharey=False, corner=True)
g.map_lower(sns.scatterplot)
g.map_diag(sns.distplot, kde=False)


# ## Second dataset

# In[11]:


g = sns.PairGrid(df_prod[features2].sample(frac=0.6), diag_sharey=False, corner=True)
g.map_lower(sns.scatterplot)
g.map_diag(sns.distplot, kde=False)


# ### Set up for modeling

# In[52]:


#data without production company
X_og = df_og[features1]
y_og = df_og[target]

#data including production company
X_prod = df_prod[features2]
y_prod = df_prod[target]


# In[53]:


#Split the first dataset 55 - 25 - 20 train/val/test

X_og_80, X_og_test, y_og_80, y_og_test = train_test_split(X_og, y_og, test_size=0.2,random_state=42)
X_og_train, X_og_val, y_og_train, y_og_val = train_test_split(X_og_80, y_og_80, test_size=.25, random_state=43)


# In[54]:


#Split the second data 55 - 25 - 20 train/val/test

X_prod_80, X_prod_test, y_prod_80, y_prod_test = train_test_split(X_prod, y_prod, test_size=0.2,random_state=42)
X_prod_train, X_prod_val, y_prod_train, y_prod_val = train_test_split(X_prod_80, y_prod_80, test_size=.25, random_state=43)


# ## Check Lasso Regularization

# In[55]:


#data without production company
lasso_model_og = Lasso()

lasso_model_og.fit(X_og_train, y_og_train)


# In[56]:


lasso_model_og.score(X_og_train, y_og_train)


# In[57]:


lasso_model_og.score(X_og_val, y_og_val)


# In[76]:


list(zip(features1, lasso_model_og.coef_))


# In[58]:


#data including production company
lasso_model_prod = Lasso()

lasso_model_prod.fit(X_prod_train, y_prod_train)


# In[59]:


lasso_model_prod.score(X_prod_train, y_prod_train)


# In[60]:


lasso_model_prod.score(X_prod_val, y_prod_val)


# In[75]:


list(zip(features2, lasso_model_prod.coef_))


# ### looks like it wants to set a lot of these equal to zero, oh no.
# 
# ## Check Ridge regularization now

# In[62]:


lr_model_ridge_og = Ridge()
lr_model_ridge_og.fit(X_og_train, y_og_train)


# In[63]:


#data without production company
lr_model_ridge_og.score(X_og_train, y_og_train)


# In[64]:


lr_model_ridge_og.score(X_og_val, y_og_val)


# In[65]:


list(zip(X_og_train, lr_model_ridge_og.coef_))


# In[66]:


#data with production company
lr_model_ridge_prod = Ridge()
lr_model_ridge_prod.fit(X_prod_train, y_prod_train)


# In[67]:


lr_model_ridge_prod.score(X_prod_train, y_prod_train)


# In[68]:


lr_model_ridge_prod.score(X_prod_val, y_prod_val)


# ### Wow so ridge has way higher R^2 scores! 
# 
# ### Gonna check out stats values next

# In[70]:


sm.add_constant(X_og_train).head()


# ## Check the stats for model without production company

# In[72]:


import statsmodels.api as sm
import statsmodels.formula.api as smf 

model_og = sm.OLS(y_og_train, sm.add_constant(X_og_train))
results = model_og.fit()

results.summary()


# ## Check the stats for model including production company

# In[73]:


import statsmodels.api as sm
import statsmodels.formula.api as smf 

model_prod = sm.OLS(y_prod_train, sm.add_constant(X_prod_train))
results_prod = model_prod.fit()

results_prod.summary()


# ## Back to the Lasso model

# check the plot for data without prod company

# In[91]:


test_set_pred_og = lasso_model_og.predict(X_og_test)


# In[93]:


residuals = abs(test_set_pred_og - y_og_test)


# In[97]:


plt.figure(figsize = (6,6))
plt.scatter(test_set_pred_og, y_og_test, alpha = 0.2)
plt.plot(y_og_test, y_og_test, 'm--', label = 'Ideal Prediction', c = 'g')
plt.title('Actual vs. Scores', fontsize = 14)
plt.xlabel('Predicted', fontsize = 12)
plt.ylabel('Actual', fontsize = 12)
plt.legend()


# In[90]:


#r-squared
r2_score(y_og_test, test_set_pred)


# ## Probably have some variables that are colinear
# # Next try lassocv and ridgecv

# **lassocv to find the best alpha**

# In[121]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#find the mean and standard deviation of the training set to scale them
std_og = StandardScaler()
std_og.fit(X_og_train.values)


#Scale the Predictors on both the train and test set based on the scale found above
X_tr_og = std_og.transform(X_og_train.values)
X_v_og = std_og.transform(X_og_val.values)


# In[122]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha

lasso_model_og2 = LassoCV(cv=5)
lasso_model_og2.fit(X_tr_og, y_og_train)


# In[123]:


#this was the best alpha it found
lasso_model_og2.alpha_


# In[124]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_og_train.columns, lasso_model_og2.coef_))


# In[125]:


# Make predictions on the test set using the new model
val_set_pred = lasso_model_og2.predict(X_v_og)


# In[109]:


#Mean Absolute Error (MAE)
def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true)) 


# In[126]:


#find the MAE and R^2 on the test set using this model
mae(y_og_test, val_set_pred)


# In[127]:


#find the R^2 on the test set using this model
r2_score(y_og_val, val_set_pred)


# **Now apply it on the data with prod co**

# In[128]:


#find the mean and standard deviation of the training set for scaling
std_prod = StandardScaler()
std_prod.fit(X_prod_train.values)


#Scale the Predictors on both the train and test set based on the scale found above
X_tr_prod = std_prod.transform(X_prod_train.values)
X_v_prod = std_prod.transform(X_prod_val.values)


# In[129]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha

lasso_model_prod2 = LassoCV(cv=5)
lasso_model_prod2.fit(X_tr_prod, y_prod_train)


# In[130]:


#this was the best alpha it found
lasso_model_prod2.alpha_


# In[131]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_prod_train.columns, lasso_model_prod2.coef_))


# In[133]:


# Make predictions on the test set using the new model
val_set_pred = lasso_model_prod2.predict(X_v_prod)


# In[135]:


#find the MAE and R^2 on the test set using this model
mae(y_prod_val, val_set_pred)


# In[136]:


#find the R^2 on the test set using this model
r2_score(y_prod_val, val_set_pred)


# ### Try Ridgecv now

# **Try on the data without production company**

# In[137]:


#find the mean and standard deviation of the training seet
std_og = StandardScaler()
std_og.fit(X_og_train.values)


#Scale the Predictors on both the train and test set based on the scale found above
X_tr_og = std_og.transform(X_og_train.values)
X_v_og = std_og.transform(X_og_val.values)


# In[139]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha

ridge_model_og2 = RidgeCV(cv=5)
ridge_model_og2.fit(X_tr_og, y_og_train)


# In[141]:


#this was the best alpha it found
ridge_model_og2.alpha_


# In[144]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_og_train.columns, ridge_model_og2.coef_))


# In[145]:


# Make predictions on the val set using the new model
val_set_pred_og2 = ridge_model_og2.predict(X_v_og)


# In[148]:


#find the MAE and R^2 on the test set using this model
mae(y_og_val, val_set_pred_og2)


# In[149]:


#find the R^2 on the test set using this model
r2_score(y_og_val, val_set_pred_og2)


# In[ ]:





# In[ ]:





# **Try on the data including production company**

# In[138]:


#find the mean and standard deviation of the training seet
std_prod = StandardScaler()
std_prod.fit(X_prod_train.values)


#Scale the Predictors on both the train and test set based on the scale found above
X_tr_prod = std_prod.transform(X_prod_train.values)
X_v_prod = std_prod.transform(X_prod_val.values)


# In[140]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha

ridge_model_prod2 = RidgeCV(cv=5)
ridge_model_prod2.fit(X_tr_prod, y_prod_train)


# In[142]:


#this was the best alpha it found
ridge_model_prod2.alpha_


# In[143]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_prod_train.columns, ridge_model_prod2.coef_))


# In[146]:


# Make predictions on the test set using the new model
val_set_pred_prod2 = ridge_model_prod2.predict(X_v_prod)


# In[147]:


#find the MAE and R^2 on the test set using this model
mae(y_prod_val, val_set_pred_prod2)


# In[150]:


#find the R^2 on the test set using this model
r2_score(y_prod_val, val_set_pred_prod2)


# | W/ Production Co. |  Train  |  Validate  |  MAE  |   
# |-------------------|:-------:|------------|-------|
# |  **LassoCV**      | ~0.39   |  ~0.35     | ~1.8  |   
# |  **RidgeCV**      | ~0.52   |  ~0.49     | ~2.0  |     

# | W/OUT Production Co. |  Train  |  Validate  |  MAE  |   
# |----------------------|:-------:|------------|-------|
# |  **LassoCV**         | ~0.33   |  ~0.45     | ~2.4  |   
# |  **RidgeCV**         | ~0.39   |  ~0.46     | ~1.6  |   

# ## All things considered, I will choose a ridge model on the data without production company

# In[ ]:




