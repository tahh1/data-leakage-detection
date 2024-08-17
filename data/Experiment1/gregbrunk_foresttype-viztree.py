#!/usr/bin/env python
# coding: utf-8

# ## Visualizations and Decision Tree Classifications on the Forest Type Dataset

# In[1]:


# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


from IPython.display import SVG
from graphviz import Source
from IPython.display import display

import os
print((os.listdir("../input/forest-cover-type-kernels-only")))


# In[2]:


# new dataframe onthe training dataset
train = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv')
train.columns


# In[3]:


train


# In[4]:


train.shape


# In[5]:


train.describe()


# Run some visualizations to look at the data.  First, look at all the continuous variables.

# In[6]:


train[train.columns[1:11]].hist(figsize = (20,15))


# Data are usable, except that aspect is a cardinal direction and may need some different encoding (since 0 is the same as 360).  Work on that later.  
# 
# Another version of the same chart, this time in Seaborn rather than pandas plot

# In[7]:


cols = train.columns[1:11]
fig, axes = plt.subplots(nrows = 1, ncols = len(cols), figsize = (30,5))
for i, ax in enumerate(axes):
    sns.distplot(train[cols[i]], ax=ax)
    sns.despine()


# Now that we know what is in the dataset, let's look at the distributions for each forest type (the target variable).  A violin plot will work.

# In[8]:


cols = train.columns[1:11]
fig, axes = plt.subplots(nrows = 1, ncols = len(cols), figsize = (30,5))
for i, ax in enumerate(axes):
    sns.violinplot(data=train, x = "Cover_Type", y = cols[i],ax=ax)
    sns.despine()
plt.tight_layout()


# Elevation distinguishes the cover types faily well.  Aspect has some bimodal distributions (due to the data encoding problem stated before), other features show smaller variations.
# 
# Now let's look at the categorical features.  Wilderness area and soil type are already one-hot encoded in the dataset.  For ploting convenience, create new columns that combine the categories back together.

# In[9]:


areas_list  = [ 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']
train['Wilderness_Area'] = train.Wilderness_Area1 * 1 + train.Wilderness_Area2 * 2 + train.Wilderness_Area3 * 3 + train.Wilderness_Area4 *4


# In[10]:


train['Soil_Type'] = (train.Soil_Type1 * 1 + 
                    train.Soil_Type2 * 2 + 
                    train.Soil_Type3 * 3 + 
                    train.Soil_Type4 * 4 + 
                    train.Soil_Type5 * 5 + 
                    train.Soil_Type6 * 6 + 
                    train.Soil_Type7 * 7 + 
                    train.Soil_Type8 * 8 + 
                    train.Soil_Type9 * 9 + 
                    train.Soil_Type10 * 10 + 
                    train.Soil_Type11 * 11 + 
                    train.Soil_Type12 * 12 + 
                    train.Soil_Type13 * 13 + 
                    train.Soil_Type14 * 14 + 
                    train.Soil_Type15 * 15 + 
                    train.Soil_Type16 * 16 + 
                    train.Soil_Type17 * 17 + 
                    train.Soil_Type18 * 18 + 
                    train.Soil_Type19 * 19 + 
                    train.Soil_Type20 * 20 + 
                    train.Soil_Type21 * 21 + 
                    train.Soil_Type22 * 22 + 
                    train.Soil_Type23 * 23 + 
                    train.Soil_Type24 * 24 + 
                    train.Soil_Type25 * 25 + 
                    train.Soil_Type26 * 26 + 
                    train.Soil_Type27 * 27 + 
                    train.Soil_Type28 * 28 + 
                    train.Soil_Type29 * 29 + 
                    train.Soil_Type30 * 30 + 
                    train.Soil_Type31 * 31 + 
                    train.Soil_Type32 * 32 + 
                    train.Soil_Type33 * 33 + 
                    train.Soil_Type34 * 34 + 
                    train.Soil_Type35 * 35 + 
                    train.Soil_Type36 * 36 + 
                    train.Soil_Type37 * 37 + 
                    train.Soil_Type38 * 38 + 
                    train.Soil_Type39 * 39 + 
                    train.Soil_Type40 * 40)


# Now make a couple of bar charts for the categorical variables

# In[11]:


# this is a useful plot for categorical variables
cols = train.columns[-2:]
fig, axes = plt.subplots(ncols = 1, nrows = len(cols), figsize = (20,10))
for i, ax in enumerate(axes):
    sns.barplot(data=train.groupby(by = [cols[i],"Cover_Type"]).Id.count().reset_index(),
                  x=cols[i], y="Id", hue="Cover_Type", ax=ax)
    sns.despine()
plt.tight_layout()


# Just knowing the winderness type is a major way to identify the species (e.g., cover type 4 is only found in wilderness area 4).
# 
# Make a decision tree, use the one-hot encoded fields for categorical variables.

# In[12]:


# Make sure I'm getting the right columns
train.columns[1:-3]


# In[13]:


labels = train.columns[1:-3]
y = train.Cover_Type
X = train[labels]


# In[14]:


X


# Run and visualize a decision tree for fun

# In[15]:


estimator = tree.DecisionTreeClassifier(max_depth = 10)
estimator.fit(X, y)

graph = Source(tree.export_graphviz(estimator, out_file=None
   , feature_names= labels, class_names = ['0', '1', '2', '3', '4' ,'5','6']
   , filled = True))

display(SVG(graph.pipe(format='svg')))


# Cool, but not too informative.  We see that elevation is a key feature.  
# 
# Now split the data and try some classifier algorithms.

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

estimator = tree.DecisionTreeClassifier(max_depth = 5)
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# Let's see how the performance varies with the depth of the tree.

# In[17]:


depths = []
performance = []
for depth in range(2,50):
    estimator = tree.DecisionTreeClassifier(max_depth = depth)
    estimator.fit(X_train, y_train)
    correct = sum(estimator.predict(X_test)==y_test)/len(y_test)
    #print('Depth = ',depth,' correct = ',correct)
    depths.append(depth)
    performance.append(correct)
results =  pd.DataFrame()
results['tree_depths'] = depths
results['performance'] = performance
results.plot(x = 'tree_depths', y = 'performance')


# Interesting, I thought that the tree would over-fit with greater depth, but performance didn't taper significantly.  Still, a depth of 15  looks about right.  Look at the min_wieght_fraction_leaf hyperparameter.

# In[18]:


depths = []
performance = []
for depth in range(0,50):
    fraction = depth/100
    estimator = tree.DecisionTreeClassifier(min_weight_fraction_leaf = fraction)
    estimator.fit(X_train, y_train)
    correct = sum(estimator.predict(X_test)==y_test)/len(y_test)
    #print('Depth = ',depth,' correct = ',correct)
    depths.append(fraction)
    performance.append(correct)
results =  pd.DataFrame()
results['tree_depths'] = depths
results['performance'] = performance
results.plot(x = 'tree_depths', y = 'performance')


# No overfitting there!

# For a tree of depth 15, let's look at the performance closer.   How did we do for each cover type? 
# 
# Look at a confusion matrix.

# In[19]:


estimator = tree.DecisionTreeClassifier(max_depth = 15)
estimator.fit(X_train, y_train)
#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# ~76%

# In[20]:


conf_mx = confusion_matrix(y_test, estimator.predict(X_test))

ax = sns.heatmap(conf_mx, annot = True, fmt = 'd')
ax.set(xlabel='Predicted', ylabel='Actual')


# Zero out the diagonals to make the colors pop, and present as a fraction of each actual type.

# In[21]:


row_sums = conf_mx.sum(axis=1, keepdims = True)
norm_conf_mx = conf_mx/row_sums
np.fill_diagonal(norm_conf_mx, 0)
ax = sns.heatmap(norm_conf_mx, annot = True)#, fmt = 'd')
ax.set(xlabel='Predicted', ylabel='Actual')


# We may want to target feature engineering to splitting out the types that are presenting a problem.   
# The next steps: 
# 1. work on the aspect feature
# 2. work on scaling
# 3. work on other features
# 4. compare to other algorithms
# 5. play with decision rationales and percent probabilities.

# Let's at least take care of the aspect data - let's convert degrees into unit circle Xs and Ys - two features, same information.

# In[22]:


# test my formula - use np instead of math
for deg in range(0,370,30):
    print((deg, np.sin(deg*np.pi/180),np.cos(deg*np.pi/180)))


# In[23]:


train['Aspect_N_S'] = np.cos(train.Aspect*np.pi/180)
train['Aspect_E_W'] = np.sin(train.Aspect*np.pi/180)
train[['Aspect', 'Aspect_N_S', 'Aspect_E_W']]


# In[24]:


# new column names of interest.  
X_col_names = [train.columns[1]]+train.columns[3:-5].tolist()+train.columns[-2:].tolist()
X_col_names


# In[25]:


y = train.Cover_Type
X = train[X_col_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

estimator = tree.DecisionTreeClassifier(max_depth = 15)
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# ~77%  A little better.  Now try with a standard scaler.

# In[26]:


scaler = StandardScaler()


estimator = tree.DecisionTreeClassifier(max_depth = 15)
estimator.fit(scaler.fit_transform(X_train), y_train)

#calculate the percent correct
sum(estimator.predict(scaler.transform(X_test))==y_test)/len(y_test)


# Not different, but I don't think we would expect a decision tree to give different results.  
# 
# Now try logistic regression before and after scaling.

# In[27]:


# logistic regression code for comparison
estimator = LogisticRegression()
estimator.fit(X_train, y_train)

#calculate the percent correct
print(('unscaled = ',sum(estimator.predict(X_test)==y_test)/len(y_test)))

# logistic regression code for comparison
estimator = LogisticRegression()
estimator.fit(scaler.fit_transform(X_train), y_train)

#calculate the percent correct
print(('scaled = ',sum(estimator.predict(scaler.transform(X_test))==y_test)/len(y_test)))


# Not much different and not too good.  
# 
# Try random forest
# 

# In[28]:


estimator = RandomForestClassifier()
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# Magically up to 81%. . . let's look at feature importance

# In[29]:


pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], data = estimator.feature_importances_).sort_values(ascending = False,by='feature_importance')


# Experiment with logisitic regression for the continuous variables only

# In[30]:


# new column names of interest.  
X2_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-2:].tolist()
X2_col_names


# In[31]:


y2 = train.Cover_Type
X2 = train[X2_col_names]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=43)


# In[32]:


# logistic regression code for comparison
estimator = LogisticRegression()
estimator.fit(X2_train, y2_train)

#calculate the percent correct
print(('unscaled = ',sum(estimator.predict(X2_test)==y2_test)/len(y2_test)))

# logistic regression code for comparison
estimator = LogisticRegression()
estimator.fit(scaler.fit_transform(X2_train), y2_train)

#calculate the percent correct
print(('scaled = ',sum(estimator.predict(scaler.transform(X2_test))==y2_test)/len(y2_test)))


# Wow, super bad.  The soil type and wilderness areas really make a difference.  
# 
# One hot encoding of categorical variables might be bad for decision trees.  Let's try binary encoding to reduce the number of columns that are needed (code a new column for each place value in the binary value for the category.)

# In[33]:


wilderness_area_lookup = {}
for n in range(5):
    binstr =format(n, '03b')
    vals = [int(binstr[i]) for i in range(len(binstr))]
    wilderness_area_lookup[n] = vals
wilderness_area_lookup


# In[34]:


# looping is a slow way to do this but it is adequate 
for row in train.index:
    bin_list = wilderness_area_lookup[train.loc[row,'Wilderness_Area']]
    train.loc[row,'Wilderness_Area_bin0'] = bin_list[0]
    train.loc[row,'Wilderness_Area_bin1'] = bin_list[1]
    train.loc[row,'Wilderness_Area_bin2'] = bin_list[2]
train.head()


# In[35]:


soil_type_lookup = {}
for n in range(41):
    binstr =format(n, '06b')
    vals = [int(binstr[i]) for i in range(len(binstr))]
    soil_type_lookup[n] = vals
soil_type_lookup


# In[36]:


# looping is a slow way to do this but it is adequate 
for row in train.index:
    bin_list = soil_type_lookup[train.loc[row,'Soil_Type']]
    train.loc[row,'Soil_Type_bin0'] = bin_list[0]
    train.loc[row,'Soil_Type_bin1'] = bin_list[1]
    train.loc[row,'Soil_Type_bin2'] = bin_list[2]
    train.loc[row,'Soil_Type_bin3'] = bin_list[3]
    train.loc[row,'Soil_Type_bin4'] = bin_list[4]
    train.loc[row,'Soil_Type_bin5'] = bin_list[5]
train.head()


# In[37]:


train.columns


# In[38]:


# new column names of interest.  
X_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-11:].tolist()
X_col_names


# In[39]:


y = train.Cover_Type
X = train[X_col_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

estimator = tree.DecisionTreeClassifier(max_depth = 15)
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# In[40]:


estimator = RandomForestClassifier()
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# .... Not much different!  Consider trying the H2O random forest for another comparison.

# In[41]:


pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], data = estimator.feature_importances_).sort_values(ascending = False,by='feature_importance')


# Intuitively it seems like slope would be a multiplier on aspect.  Aspect runs from about 0 to 40, let's multiply slope by aspect.    

# In[42]:


train['Aspect_N_S_Slope'] = train['Aspect_N_S'] * train['Slope'] 
train['Aspect_E_W_Slope'] = train['Aspect_E_W'] * train['Slope'] 


# In[43]:


# new column names of interest.  
X_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-13:].tolist()
X_col_names


# In[44]:


y = train.Cover_Type
X = train[X_col_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

estimator = tree.DecisionTreeClassifier(max_depth = 15)
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# In[45]:


estimator = RandomForestClassifier()
estimator.fit(X_train, y_train)

#calculate the percent correct
sum(estimator.predict(X_test)==y_test)/len(y_test)


# In[46]:


pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], 
             data = estimator.feature_importances_
            ).sort_values(ascending = False,by='feature_importance')


# Not much different.  Try enhancing the Elevation field with the Aspect_N_S_Slope field.  Seems like an addition/ subtraction problem.  Let's iteratively try it and see if it makes a difference:

# In[47]:


factors = [0,1,2,4,6,8,10,20,100, 1000, 10]
for factor in factors:
    train['Elev_Asp_Slope'] = train['Aspect_N_S_Slope'] * factor +  train['Elevation'] 
    X_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-14:].tolist()
    y = train.Cover_Type
    X = train[X_col_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)
    estimator = RandomForestClassifier()
    estimator.fit(X_train, y_train)
    pct = sum(estimator.predict(X_test)==y_test)/len(y_test)
    print(('factor = ',factor, '; percent correct = ', pct)) 


# In[48]:


pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], data = estimator.feature_importances_).sort_values(ascending = False,by='feature_importance')


# Elev_asp_slope was used by the tree, but it didn't improve the prediction (all in the noise).  See if it helped logistic regression.  
#  

# In[49]:


# logistic regression code for comparison
estimator = LogisticRegression()
estimator.fit(scaler.fit_transform(X_train), y_train)
pct = sum(estimator.predict(scaler.transform(X_test))==y_test)/len(y_test)
pct


# Nope - even worse than before!  Let's do some cross-validation before doing hyperparameter tuning on the random forest.

# In[50]:


# Cross validation
estimator = RandomForestClassifier()
scores = cross_val_score(estimator, X_train, y_train, cv=10)

pd.Series(scores).describe()


# In[51]:


estimator.get_params()


# In[52]:


# Number of trees in random forest
n_estimators = [3, 5, 10, 50, 100]
# Number of features to consider at every split
max_features = ['auto', None]
# Maximum number of levels in tree
max_depth = [3, 5, 10, 50, 100, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[53]:


# Use the random grid to search for best hyperparameters
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
random_search = RandomizedSearchCV(estimator, param_distributions = random_grid, 
                               n_iter = 20, cv = 3, verbose=2, random_state=42)# Fit the random search model
random_search.fit(X_train, y_train)


# In[54]:


random_search.best_params_


# In[55]:


random_search.best_score_


# In[56]:


random_search.best_estimator_


# In[57]:


random_search.best_estimator_


# In[58]:


# see how it performs on the test set
pct = sum(random_search.predict(X_test)==y_test)/len(y_test)
pct


# The accuracy is creeping up with hyperparameter tuning.  On the to-do list:  
# * package the data wrangling and feature engineering as a pipeline
# * develop and tune other models (SVM, LR, etc.), ensemble with random forest
# * more feature analysis/ engineering (e.g., understand confusions)
# * start on neural nets 
