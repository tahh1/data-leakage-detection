#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reading the dataset(only for kernel!!)
import pandas as pd
import numpy as np
from collections import Counter
train_raw = pd.read_csv("../input/Train_UWu5bXk.csv")
test_raw = pd.read_csv("../input/Test_u94Q5KV.csv")


# In[2]:


### Data Exploration & Feature Engineering ###

##1. Data Exploration

#Combine test and train into one file and set source to identify test or train
train_raw['source']='train'
test_raw['source']='test'
data = pd.concat([train_raw, test_raw],ignore_index=True)
print((train_raw.shape, test_raw.shape, data.shape))

data.head()


# In[3]:


#Check missing values:
data.apply(lambda x: sum(x.isnull()))

## we’ll impute the missing values in Item_Weight and Outlet_Size in the data cleaning section.


# In[4]:


#Numerical data summary:
data.describe()

#1 Feature Engeneering:
#Item_Visibility has a min value of zero.
#This makes no practical sense because when a product is being sold in a store,the visibility cannot be 0.

#2
#Outlet_Establishment_Years vary from 1985 to 2009. The values might not be apt in this form. 
#Rather, if we can convert them to how old the particular store is


# In[5]:


#Number of unique values in each:
data.apply(lambda x: len(x.unique()))

#This tells us that there are 1559 products and 10 outlets/stores
# Another thing that should catch attention is that Item_Type has 16 unique values.


# In[6]:


#Filter categorical variables
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
#Exclude ID cols and source:
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]
#Print frequency of categories
for col in categorical_columns:
    print(('\nFrequency of Categories for varible %s'%col))
    print((data[col].value_counts()))
    
##The output gives us following observations:

##Item_Fat_Content: Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. 
##Also, some of ‘Regular’ are mentioned as ‘regular’.

##Item_Type: Not all categories have substantial numbers. 
##It looks like combining them can give better results.


# In[7]:


## First, we need to understand the distriution of Item_Weight. We can understand it better,
## if we can visually see it. Here, we will plot the histogram.
get_ipython().run_line_magic('matplotlib', 'inline')
## inline matplotlib command

## plot before imputing
data.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(10,6), title='Histogram of Item_Weight')


# In[8]:


### Data Cleaning ###

#Determine the average weight per item:
item_avg_weight = data.groupby('Item_Identifier').Item_Weight.mean()

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
print(('Orignal #missing: %d'% sum(miss_bool)))

## replace na values with the average of the item_weight for that particular product
data.Item_Weight.fillna(0, inplace = True)
for index, row in data.iterrows():
    if(row.Item_Weight == 0):
        data.loc[index, 'Item_Weight'] = item_avg_weight[row.Item_Identifier]
        #print(item_avg_weight[row.Item_Identifier])

print(('Final #missing: %d'% sum(data['Item_Weight'].isnull())))


# In[9]:


## plot after imputing
data.Item_Weight.plot(kind='hist', color='white', edgecolor='black', figsize=(10,6), title='Histogram of Item_Weight')

## we didnt create too much bias so this imputation is viable


# In[10]:


data.groupby('Outlet_Identifier').Outlet_Size.value_counts(dropna=False)
## see that only OUT010, OUT017, OUT045 HAS NA values


# In[11]:


data.groupby('Outlet_Type').Outlet_Size.value_counts(dropna=False)
## we notice that :
## grocery store and supermarket Type 1 has only small as the outlet size
## so we can replace the nan with small


# In[12]:


data.loc[data.Outlet_Identifier.isin(['OUT010','OUT017','OUT045']), 'Outlet_Size'] = 'Small'


# In[13]:


data.Outlet_Size.value_counts()


# In[14]:


### Feature Engineering ###

data.min()


# In[15]:


## Notice that Item_Visibility has a minimum value of 0. It seems absurd that an item has 0 
## visibility. Therefore, we will modify that column.
## Here we Group by Item_Identifier, calculate mean for each group(excluding zero values), then we proceed
## to replace the zero values in each group with the group's mean.

## we have to replace 0's by na because, mean() doesnt support exclude '0' parameter 
##but it includes exclude nan parameter which is true by default

data.loc[data.Item_Visibility == 0, 'Item_Visibility'] = np.nan

#aggregate by Item_Identifier
IV_mean = data.groupby('Item_Identifier').Item_Visibility.mean()
IV_mean


# In[16]:


data.Item_Visibility.fillna(0, inplace=True)

#replace 0 values
for index, row in data.iterrows():
    if(row.Item_Visibility == 0):
        data.loc[index, 'Item_Visibility'] = IV_mean[row.Item_Identifier]
        #print(combined.loc[index, 'Item_Visibility'])
        
data.Item_Visibility.describe()
## see that min value is not zero anymore


# In[17]:


#Create a broad category of Type of Item

#Get the first two characters of ID:
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
#Rename them to more intuitive categories:
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[18]:


#Determine the years of operation of a store
#Years:
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()


# In[19]:


data['MRP_Factor'] = pd.cut(data.Item_MRP, [0,70,130,201,400], labels=['Low', 'Medium', 'High', 'Very High'])


# In[20]:


#Modify categories of Item_Fat_Content

#Change categories of low fat:
print ('Original Categories:')
print((data['Item_Fat_Content'].value_counts()))

print ('\nModified Categories:')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                             'reg':'Regular',
                                                             'low fat':'Low Fat'})
print((data['Item_Fat_Content'].value_counts()))


# In[21]:


#Mark non-consumables as separate category in low_fat:
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
data['Item_Fat_Content'].value_counts()


# In[22]:


#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet', 'MRP_Factor']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])


# In[23]:


#One Hot Coding: dummy varriables

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet', 'MRP_Factor'])


# In[24]:


data.dtypes
#Here we can see that all variables are now float and each category has a new variable.


# In[37]:


data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)


# In[26]:


### Exporting Data ###
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year',],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
#train.to_csv("train_modified.csv",index=False)
#test.to_csv("test_modified.csv",index=False)


# In[27]:


train.head()


# In[28]:


test.head()


# In[29]:


## lets draw some plots to see that the regression assumptions are not voilated
## QQ plot

import pylab 
import scipy.stats as stats

quantile = train.Item_Outlet_Sales

stats.probplot(quantile, dist="uniform", plot=pylab)
pylab.show()

## the line is almost linear except for the end points 


# In[30]:


### Model Building ###

#Define target and ID columns:

##Since I’ll be making many models, instead of repeating the codes again and again, 
##I would like to define a generic function which takes the algorithm and data as input and makes the model
##performs cross-validation and generates submission

# we want to predict target
target = 'Item_Outlet_Sales'

#below are just identifiers which we dont want to fit
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn import metrics
from sklearn.model_selection import cross_validate, cross_val_score
import matplotlib.pyplot as plt

def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename, resid=False, transform=False):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    residuals = dtrain_predictions - dtrain[target]
    if(transform == True):
        train_mod = train.copy(deep = True)
        train_mod[target] = train_mod[target].apply(np.log)
        dtrain_predictions = np.exp(dtrain_predictions)
        #print(dtrain_predictions)

    
    #residuals vs fitted plot
    if(resid == True):
        plt.scatter(dtrain_predictions, residuals)
        plt.xlabel('fitted values')
        plt.ylabel('residuals')
        plt.show()
    
    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print(("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions))))
    print(("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)
    


# In[31]:


### Linear Regression Model

from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
pred1 = np.nan
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv', resid=True)


coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients', figsize=(10,6))

#if you notice the coefficients, they are very large in magnitude which signifies overfitting. 
#To cater to this, we will use a ridge regression model.

## residual vs fitted plot and model coefficients plot is given below


# In[32]:


## As the residual vs fitted plot is funnel shaped, 
## the response variable suffers from non-constant variance
##we can do a log transformation, square root trasformation on the response variable
## to make it linear and to improve the model even further


# In[33]:


## Ridge Regression Model:

## lets take alpha 0.05 for now for both Ridge and Lasso

alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')

#The regression coefficient got better now, also the cross validation score has improved
##bot the rmse didnt change much


# In[34]:


## Lasso Regression Model:

alg3 = Lasso(alpha=0.05,normalize=True)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.coef_, predictors).sort_values()
coef3.plot(kind='bar', title='Model Coefficients', figsize=(10,6))

## you can see that the coefficients of some columns have decreased but for some variables the 
## coeffiecients has almost doubled than that of Ridge regression
## also the mean cross validation score has increased in comparison to Ridge
## RMSE didnt change much


# In[36]:


## comparing the cross validation score for all three models Ridge regression has the lowest mean score
## and lowest model coefficients for all columns
## RMSE for all the models were almost same

## you can run a 'for' loop for alpha between 1 and 20 to see if the model improves
## the cross validation and other metrics and choose the best model 

