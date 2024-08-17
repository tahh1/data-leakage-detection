#!/usr/bin/env python
# coding: utf-8

# ## I will be using the dataset that has the dummy variables for genre and has the ordinal data for MPAA rating. (same as in the third EDA notebook)
# 
# I will first do a Train-Validate-Test split, and (if I have time) a K-fold cross validation. I will apply these on a linear regression, polynomial, and Ridge regression model. Then I will choose the best one.

# In[35]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge #ordinary linear regression + w/ ridge regularization
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# In[36]:


# Load data
df = pd.read_pickle('Ordinal_MPAA_merged_with_dummy_genres.pkl')


# In[37]:


#These are the only columns we care about for our model
features = ['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating',
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
            'History', 'Horror', 'Music', 'Mystery', 'Other', 'Romance', 
            'Sci-fi', 'Sport', 'Thriller', 'War', 'Ordinal_MPAA']

target = 'Num_of_Seasons'


# ## Train-Validation-Test Split

# In[38]:


#split off the test set

X = df[features]
y = df[target]

#hold out 20% of the data for final testing

X, X_test, y, y_test = train_test_split(X, y, test_size = .2, random_state = 10)


# In[39]:


#split the remaining data into train and validation

#keep 25% for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = .25, random_state = 3)


# ### Setting up the models I'm going to use
# 
# - Going to standardize the data for regulariztion
# - Get some polynomial features for the poly model

# In[40]:


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


# In[41]:


#Train
lm.fit(X_train, y_train)
lm_reg.fit(X_train_scaled, y_train)
lm_poly.fit(X_train_poly, y_train)


# lm.score(X_train, y_train)

# In[43]:


#validate
print(('basic regression R^2', lm.score(X_valid, y_valid)))
print(('Ridge regression R^2', lm_reg.score(X_valid_scaled, y_valid)))
print(('Poly regression R^2', lm_poly.score(X_valid_poly, y_valid)))


# Looks like poly regression is overfitting. 
# 
# It also seems like basic regression and Ridge regression have the same R^2. I'll since basic regression score is slightly higher, I will choose to train that one on the entire training+validation set and then check my R^2 with the test set.
# 
# Further, I can change the seed and rerun the same analyses to double check which might be better

# In[44]:


#recall that X and y have the train+validate sets already
lm.fit(X, y)
lm.score(X_test, y_test)


# Omg how did it do so poorly now! This probably means my model is very overfit...

# ## Try a different seed

# In[45]:


#split off the test set

X2 = df[features]
y2 = df[target]

#hold out 20% of the data for final testing

X2, X2_test, y2, y2_test = train_test_split(X2, y2, test_size = .2, random_state = 42)


# In[46]:


#split the remaining data into train and validation

#keep 25% for validation
X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2, y2, test_size = .25, random_state = 3)


# In[47]:


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


# In[48]:


#Train
lm2.fit(X2_train, y2_train)
lm2_reg.fit(X2_train_scaled, y2_train)
lm2_poly.fit(X2_train_poly, y2_train)


# In[49]:


lm2.score(X2_train, y2_train)


# In[50]:


#validate
print(('basic regression R^2', lm2.score(X2_valid, y2_valid)))
print(('Ridge regression R^2', lm2_reg.score(X2_valid_scaled, y2_valid)))
print(('Poly regression R^2', lm2_poly.score(X2_valid_poly, y2_valid)))


# hmm, the scores went down, and it looks like poly regression might not have overfit this time.. I'll retrain them and then test basic regression and maybe double check the others

# In[51]:


#recall that X and y have the train+validate sets already
lm2.fit(X2, y2)
lm2.score(X2_test, y2_test)


# wuuuuuuuut... it did better this time! What's with all this variabilty... 

# In[28]:


#i"m just gonna test the other two as well
lm2_reg.fit(X2, y2)
lm2_reg.score(X2_test, y2_test)


# In[29]:


lm2_poly.fit(X2, y2)
lm2_poly.score(X2_test, y2_test)


# bruh..

# ## Try another seed again

# In[62]:


#split off the test set

X3 = df[features]
y3 = df[target]

#hold out 20% of the data for final testing

X3, X3_test, y3, y3_test = train_test_split(X3, y3, test_size = .2, random_state = 14)


# In[63]:


#split the remaining data into train and validation

#keep 25% for validation
X3_train, X3_valid, y3_train, y3_valid = train_test_split(X3, y3, test_size = .25, random_state = 3)


# In[64]:


#set up the 3 models we're choosing from:

#basic regression
lm3 = LinearRegression()

#Feature scaling for train, val, and test so that we can run our ridge model on each
scaler = StandardScaler()

X3_train_scaled = scaler.fit_transform(X3_train.values)
X3_valid_scaled = scaler.transform(X3_valid.values)
X3_test_scaled = scaler.transform(X3_test.values)

#ridge regression
lm3_reg = Ridge(alpha=1)

#Feature transforms for train, val, and test so that we can run our poly model on each
#poly regression
poly = PolynomialFeatures(degree=2) 

X3_train_poly = poly.fit_transform(X3_train.values)
X3_valid_poly = poly.transform(X3_valid.values)
X3_test_poly = poly.transform(X3_test.values)

lm3_poly = LinearRegression()


# In[65]:


#Train
lm3.fit(X3_train, y3_train)
lm3_reg.fit(X3_train_scaled, y3_train)
lm3_poly.fit(X3_train_poly, y3_train)


# In[66]:


#print training scores
print('Training scores')
print(('basic regression R^2', lm3.score(X3_train, y3_train)))
print(('Ridge regression R^2', lm3_reg.score(X3_train_scaled, y3_train)))
print(('Poly regression R^2', lm3_poly.score(X3_train_poly, y3_train)))


# In[67]:


#validate
print('validation scores')
print(('basic regression R^2', lm3.score(X3_valid, y3_valid)))
print(('Ridge regression R^2', lm3_reg.score(X3_valid_scaled, y3_valid)))
print(('Poly regression R^2', lm3_poly.score(X3_valid_poly, y3_valid)))


# In[68]:


lm3_reg.fit(X3, y3)
lm3_reg.score(X3_test, y3_test)


# ## Try cross validation
# 
# This might be a better esp since I don't have that much data

# In[69]:


from sklearn.model_selection import KFold

X = df[features]
y = df[target]

#hold out 20% of the data for final testing

X, X_test, y, y_test = train_test_split(X, y, test_size = .2, random_state = 10)

#this helps with the way kf generates indices
X, y = np.array(X), np.array(y)


# In[31]:


from sklearn.model_selection import cross_val_score
lm = LinearRegression()

cross_val_score(lm, X, y, # estimator, features, target
                cv=5, # number of folds 
                scoring='r2') # scoring metric


# In[32]:


kf = KFold(n_splits=5, shuffle=True, random_state = 71)
cross_val_score(lm, X, y, cv=kf, scoring='r2')


# ## above is the less manual way, gonna try it manually here

# In[70]:


X4 = df[features]
y4 = df[target]

#hold out 20% of the data for final testing

X4, X4_test, y4, y4_test = train_test_split(X4, y4, test_size = .2, random_state = 15)

#this helps with the way kf generates indices
X4, y4 = np.array(X4), np.array(y4)


# In[72]:


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


# In[74]:


#ridge model did slightly better but it is about the same
#let's check on our test set now
X_scaled = scaler.fit_transform(X4)
X_test_scaled = scaler.transform(X4_test)

lm_reg = Ridge(alpha=1)
lm_reg.fit(X_scaled,y4)
print(f'Ridge Regression test R^2: {lm_reg.score(X_test_scaled, y4_test):.3f}')


# In[ ]:




