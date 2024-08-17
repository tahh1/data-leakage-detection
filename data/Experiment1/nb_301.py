#!/usr/bin/env python
# coding: utf-8

# ## I have chosen a Ridge model, with alpha = 10, on the data that does not include production company.

# In[31]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error as mae, mean_squared_error as rmse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(context='notebook', style='whitegrid', font_scale=1.2)


# In[2]:


df = pd.read_pickle('Ordinal_MPAA_merged_with_dummy_genres.pkl')


# ### Check out the relationship between number of seasons and season 1 rating

# In[203]:


fig, ax = plt.subplots()
ax.scatter(df['Season_1_Rating'], df['Num_of_Seasons'])
ax.xaxis.set_tick_params(labelsize=13)
ax.yaxis.set_tick_params(labelsize=13)
ax.set_xlabel('Season 1 Rating', fontsize = 15)
ax.set_ylabel('Number of Seasons', fontsize = 15)
ax.set_title('Rating vs. Number of Seasons', fontsize = 15)
plt.savefig('season1ratingVSseasons.png');


# In[54]:


df.sort_values(by = 'Num_of_Seasons', ascending = False).head(20)


# In[4]:


df.info()


# In[170]:


#filter the datasets to just the variables we care about, keeping target at the end
features = ['Start_Year', 'Num_Episodes_Per_Season', 'Season_1_Rating',
            'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 
            'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
            'History', 'Horror', 'Music', 'Mystery', 'Other', 'Romance', 
            'Sci-fi', 'Sport', 'Thriller', 'War', 'Ordinal_MPAA']

target = 'Num_of_Seasons'


# In[171]:


X = df[features]
y = df[target]


# ### Split into train-validate-test

# In[172]:


#Split the data 55 - 25 - 20 train/val/test

X_80, X_test, y_80, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_80, y_80, test_size=.25, random_state=43)


# ### Scale my variables

# In[173]:


#find the mean and standard deviation of the training seet
std = StandardScaler()
std.fit(X_train.values)


#Scale the Predictors on both the train and test set based on the scale found above
X_tr = std.transform(X_train.values)
X_v = std.transform(X_val.values)


# ### Apply RidgeCV and get model

# In[174]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha

ridge_model = RidgeCV(cv=5)
ridge_model.fit(X_tr, y_train)


# In[175]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_train.columns, ridge_model.coef_))


# In[176]:


# Make predictions on the val set using the new model
val_set_pred = ridge_model.predict(X_v)


# In[177]:


#find the MAE and R^2 on the validation set using this model
mae(y_val, val_set_pred)


# In[178]:


#find the R^2 on the validation set using this model
r2_score(y_val, val_set_pred)


# ### Retrain model on train+val data, then get test scores

# In[179]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha
ridge_model = RidgeCV(cv=5)
ridge_model.fit(X_80, y_80)


# In[180]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_80.columns, ridge_model.coef_))


# ### Apply model to test set

# In[181]:


# Make predictions on the val set using the new model
val_set_pred = ridge_model.predict(X_test)


# In[182]:


mae(y_test, val_set_pred)


# In[183]:


rmse(y_test, val_set_pred)


# In[184]:


#find the R^2 on the test set using this model
r2_score(y_test, val_set_pred)


# In[185]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X_test.columns, ridge_model.coef_))


# In[186]:


sns.residplot(x=ridge_model.predict(X_test), y=y_test, lowess=True)
#plt.title('Residuals')
plt.xlabel('Predicted (Number of Seasons)')
plt.ylabel('Residuals');


# In[196]:


from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(ridge_model, train_alpha = .65, test_alpha = .65)

visualizer.fit(X_80, y_80)
visualizer.score(X_test, y_test)
#visualizer.show()
visualizer.show(outpath="yellowbrickresiduals.png");


# ### Try log y

# In[67]:


df['log_y'] = np.log(df['Num_of_Seasons'])


# In[69]:


df.head(3)


# In[72]:


y2 = df['log_y']

#Split the data 55 - 25 - 20 train/val/test

X2_80, X2_test, y2_80, y2_test = train_test_split(X, y2, test_size=0.2,random_state=42)


# In[83]:


#find the mean and standard deviation of the training seet
std2 = StandardScaler()
std2.fit(X2_80.values)


#Scale the Predictors on both the train and test set based on the scale found above
X2_tr = std2.transform(X2_80.values)


# In[84]:


#Run the cross validation, find the best alpha, refit the model on all the data with that alpha

ridge_model2 = RidgeCV(cv=5)
ridge_model2.fit(X2_tr, y2_80)


# In[85]:


#these are the standardized coeffs when the model refit using the new alpha
list(zip(X2_80.columns, ridge_model2.coef_))


# In[86]:


# Make predictions on the val set using the new model
val_set_pred2 = ridge_model2.predict(X2_test)


# In[87]:


mae(y2_test, val_set_pred2)


# In[90]:


rmse(y2_test, val_set_pred2)


# In[94]:


visualizer2 = ResidualsPlot(ridge_model2
                           )

visualizer2.fit(X2_80, y2_80)
visualizer2.score(X2_test, y2_test)
visualizer2.show();


# bruh.. log(y) is wild. I'm not gonna use that shi

# ## Can't find a summary stats for ridgecv adjusted R^2 so i'm gonna code one

# In[106]:


ridge_model.get_params()


# In[107]:


def adjusted_r2(r2, n, k):
    '''
    Function to calculate adjusted R^2
    
    <Input> r2 (Type = float): R^2, 
            n (Type = int): number of data points in the data sample
            k (Type = int): number of independent variables in th model
            
    <Returns> adjusted R^2 (Type = float)
            
    '''
    
    return (1 - ((1-r2)*(n-1)/(n-k-1)))


# In[108]:


X.shape #tells us 849 data points, 24 features


# In[109]:


adjusted_r2(r2_score(y_test, val_set_pred), 849, 24)


# In[124]:


type(ridge_model2.coef_)


# In[166]:


ridge_model2.coef_.shape


# ### Predict the lifespan of my favorite show

# In[205]:


Harley_quinn = np.array([2019,13, 8.3, 1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 6])


# In[206]:


Harley_quinn.shape


# In[210]:


print((ridge_model.predict(Harley_quinn.reshape(1,-1))))


# In[ ]:




