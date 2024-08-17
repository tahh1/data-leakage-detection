#!/usr/bin/env python
# coding: utf-8

# ## I will be using the dataset that has all genres as dummy variables, MPAA rating as ordinal data, and a binary column of whether or not a show is produced by a large company. (same as fourth eda notebook)
# 
# I will first do a Train-Validate-Test split, and (if I have time) a K-fold cross validation. I will apply these on a linear regression, polynomial, and Ridge regression model. Then I will choose the best one.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge #ordinary linear regression + w/ ridge regularization
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# In[2]:


# Load data
df = pd.read_pickle('all_scraped_features_df.pkl')


# In[3]:


#These are the only columns we care about for our model
features = ['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating',
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
            'History', 'Horror', 'Music', 'Mystery', 'Other', 'Romance', 
            'Sci-fi', 'Sport', 'Thriller', 'War', 'Ordinal_MPAA', 'Large_prod_co']

target = 'Num_of_Seasons'


# In[10]:


#split off the test set

X = df[features]
y = df[target]

#hold out 20% of the data for final testing

X, X_test, y, y_test = train_test_split(X, y, test_size = .2, random_state = 9)


# In[11]:


#split the remaining data into train and validation

#keep 25% for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = .25, random_state = 3)


# ### Setting up the models I'm going to use
# 
# - Going to standardize the data for regulariztion
# - Get some polynomial features for the poly model

# In[12]:


#set up the 3 models we're choosing from:

#basic regression
lm = LinearRegression()

#Feature scaling for train, val, and test so that we can run our ridge model on each
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.values)
X_valid_scaled = scaler.transform(X_valid.values)
X_test_scaled = scaler.transform(X_test.values)

#ridge regression
lm_reg = Ridge(alpha=1)

#Feature transforms for train, val, and test so that we can run our poly model on each
#poly regression
poly = PolynomialFeatures(degree=2) 

X_train_poly = poly.fit_transform(X_train.values)
X_valid_poly = poly.transform(X_valid.values)
X_test_poly = poly.transform(X_test.values)

lm_poly = LinearRegression()


# In[13]:


#Train
lm.fit(X_train, y_train)
lm_reg.fit(X_train_scaled, y_train)
lm_poly.fit(X_train_poly, y_train)


# In[14]:


lm.score(X_train, y_train)


# In[15]:


#validate
print(('basic regression R^2', lm.score(X_valid, y_valid)))
print(('Ridge regression R^2', lm_reg.score(X_valid_scaled, y_valid)))
print(('Poly regression R^2', lm_poly.score(X_valid_poly, y_valid)))


# Looks like poly reg is HUGELY overfitting

# In[16]:


#recall that X and y have the train+validate sets already
lm.fit(X, y)
lm.score(X_test, y_test)


# ### Try another seed

# In[17]:


#split off the test set

X2 = df[features]
y2 = df[target]

#hold out 20% of the data for final testing

X2, X2_test, y2, y2_test = train_test_split(X2, y2, test_size = .2, random_state = 42)


# In[18]:


#split the remaining data into train and validation

#keep 25% for validation
X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2, y2, test_size = .25, random_state = 3)


# In[19]:


#set up the 3 models we're choosing from:

#basic regression
lm2 = LinearRegression()

#Feature scaling for train, val, and test so that we can run our ridge model on each
scaler = StandardScaler()

X2_train_scaled = scaler.fit_transform(X2_train.values)
X2_valid_scaled = scaler.transform(X2_valid.values)
X2_test_scaled = scaler.transform(X2_test.values)

#ridge regression
lm2_reg = Ridge(alpha=1)

#Feature transforms for train, val, and test so that we can run our poly model on each
#poly regression
poly = PolynomialFeatures(degree=2) 

X2_train_poly = poly.fit_transform(X2_train.values)
X2_valid_poly = poly.transform(X2_valid.values)
X2_test_poly = poly.transform(X2_test.values)

lm2_poly = LinearRegression()


# In[20]:


#Train
lm2.fit(X2_train, y2_train)
lm2_reg.fit(X2_train_scaled, y2_train)
lm2_poly.fit(X2_train_poly, y2_train)


# In[21]:


lm2.score(X2_train, y2_train)


# In[22]:


#validate
print(('basic regression R^2', lm2.score(X2_valid, y2_valid)))
print(('Ridge regression R^2', lm2_reg.score(X2_valid_scaled, y2_valid)))
print(('Poly regression R^2', lm2_poly.score(X2_valid_poly, y2_valid)))


# In[23]:


#recall that X and y have the train+validate sets already
lm2.fit(X2, y2)
lm2.score(X2_test, y2_test)


# ## Try cross validation

# In[27]:


X4 = df[features]
y4 = df[target]

#hold out 20% of the data for final testing

X4, X4_test, y4, y4_test = train_test_split(X4, y4, test_size = .2, random_state = 15)

#this helps with the way kf generates indices
X4, y4 = np.array(X4), np.array(y4)


# In[28]:


#run the CV

kf = KFold(n_splits=5, shuffle=True, random_state = 71)
cv_lm_r2s, cv_lm_reg_r2s = [], [] #collect the validation results for both models

for train_ind, val_ind in kf.split(X4,y4):
    
    X_train, y_train = X4[train_ind], y4[train_ind]
    X_val, y_val = X4[val_ind], y4[val_ind] 
    
    #simple linear regression
    lm = LinearRegression()
    lm_reg = Ridge(alpha=1)

    lm.fit(X_train, y_train)
    cv_lm_r2s.append(lm.score(X_val, y_val))
    
    #ridge with feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lm_reg.fit(X_train_scaled, y_train)
    cv_lm_reg_r2s.append(lm_reg.score(X_val_scaled, y_val))

print(('Simple regression scores: ', cv_lm_r2s))
print(('Ridge scores: ', cv_lm_reg_r2s, '\n'))

print(f'Simple mean cv r^2: {np.mean(cv_lm_r2s):.3f} +- {np.std(cv_lm_r2s):.3f}')
print(f'Ridge mean cv r^2: {np.mean(cv_lm_reg_r2s):.3f} +- {np.std(cv_lm_reg_r2s):.3f}')


# In[29]:


#ridge model did slightly better but it is about the same
#let's check on our test set now
X_scaled = scaler.fit_transform(X4)
X_test_scaled = scaler.transform(X4_test)

lm_reg = Ridge(alpha=1)
lm_reg.fit(X_scaled,y4)
print(f'Ridge Regression test R^2: {lm_reg.score(X_test_scaled, y4_test):.3f}')


# ### Try a different seed again

# In[32]:


X5 = df[features]
y5 = df[target]

#hold out 20% of the data for final testing

X5, X5_test, y5, y_test = train_test_split(X5, y5, test_size = .2, random_state = 36)

#this helps with the way kf generates indices
X5, y5 = np.array(X5), np.array(y5)


# In[33]:


#run the CV

kf = KFold(n_splits=5, shuffle=True, random_state = 71)
cv_lm_r2s, cv_lm_reg_r2s = [], [] #collect the validation results for both models

for train_ind, val_ind in kf.split(X5,y5):
    
    X_train, y_train = X5[train_ind], y5[train_ind]
    X_val, y_val = X5[val_ind], y5[val_ind] 
    
    #simple linear regression
    lm = LinearRegression()
    lm_reg = Ridge(alpha=1)

    lm.fit(X_train, y_train)
    cv_lm_r2s.append(lm.score(X_val, y_val))
    
    #ridge with feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lm_reg.fit(X_train_scaled, y_train)
    cv_lm_reg_r2s.append(lm_reg.score(X_val_scaled, y_val))

print(('Simple regression scores: ', cv_lm_r2s))
print(('Ridge scores: ', cv_lm_reg_r2s, '\n'))

print(f'Simple mean cv r^2: {np.mean(cv_lm_r2s):.3f} +- {np.std(cv_lm_r2s):.3f}')
print(f'Ridge mean cv r^2: {np.mean(cv_lm_reg_r2s):.3f} +- {np.std(cv_lm_reg_r2s):.3f}')


# In[35]:


#ridge model did slightly better but it is about the same
#let's check on our test set now
X_scaled = scaler.fit_transform(X5)
X_test_scaled = scaler.transform(X5_test)

lm_reg = Ridge(alpha=1)
lm_reg.fit(X_scaled,y5)
print(f'Ridge Regression test R^2: {lm_reg.score(X_test_scaled, y_test):.3f}')


# ## That's a huge variance though. Gonna try Lasso

# In[ ]:




