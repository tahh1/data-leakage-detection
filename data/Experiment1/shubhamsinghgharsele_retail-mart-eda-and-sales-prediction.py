#!/usr/bin/env python
# coding: utf-8

# **Mart SALES PREDICTION**

# ![Super Market](https://www.supermarketnews.com/sites/supermarketnews.com/files/styles/article_featured_standard/public/Farmstead-2promo_0.gif?itok=LBoG0Kn8)

# # **Understanding Data**
# 
# * Item Identifier: A code provided for the item of sale
# * Item Weight: Weight of item
# * Item Fat Content: A categorical column of how much fat is present in the item : ‘Low Fat’, ‘Regular’, ‘low fat’, ‘LF’, ‘reg’
# * Item Visibility: Numeric value for how visible the item is
# * Item Type: What category does the item belong to: ‘Dairy’, ‘Soft Drinks’, ‘Meat’, ‘Fruits and Vegetables’, ‘Household’, ‘Baking Goods’, ‘Snack Foods’, ‘Frozen Foods’, ‘Breakfast’, ’Health and Hygiene’, ‘Hard Drinks’, ‘Canned’, ‘Breads’, ‘Starchy Foods’, ‘Others’, ‘Seafood’.
# * Item MRP: The MRP price of item
# * Outlet Identifier: Which outlet was the item sold. This will be categorical column
# * Outlet Establishment Year: Which year was the outlet established
# * Outlet Size: A categorical column to explain size of outlet: ‘Medium’, ‘High’, ‘Small’.
# * Outlet Location Type: A categorical column to describe the location of the outlet: ‘Tier 1’, ‘Tier 2’, ‘Tier 3’
# * Outlet Type : Categorical column for type of outlet: ‘Supermarket Type1’, ‘Supermarket Type2’, ‘Supermarket Type3’, ‘Grocery Store’
# * Item Outlet Sales: The amount of sales for an item.

# # **Loading Libraries **

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Lasso, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb

from math import sqrt

import os
print((os.listdir("../input")))

# Any results you write to the current directory are saved as output.


# # **Read  the Data**

# In[2]:


data_train = pd.read_csv("../input/Train.csv")
data_test = pd.read_csv("../input/Test.csv")


# # **Sneak Peek our data..**

# In[3]:


data_train.head()


# In[4]:


data_test.head()


# # **Data from Eagle's Eye**

# In[5]:


print(("Training Data  Row : %s Column : %s " % (str(data_train.shape[0]) ,str(data_train.shape[1]))))


# In[6]:


print(("Training Data  Row : %s Column : %s " % (str(data_test.shape[0]) ,str(data_test.shape[1]))))


# In[7]:


data_train.info()


# In[8]:


data_test.info()


# # **DATA CLEANUP - Null Data**

# In[9]:


data_train.isnull().sum()


# In[10]:


data_test.isnull().sum()


# **Observations** - Item Weight and Outlet Size has missing values both in training and test dataset

# # **Let's replace**
# 
# * "Item Weight" null values with mean values of "Item Weight"
# * "Outlet_Size" with Small as most of the null values are ( Tier 3 ,Grocery Store or Tier 2 ,Supermarket Type1) where most of the values are small.
# 

# In[11]:


data_train['Item_Weight'] = data_train['Item_Weight'].fillna((data_train['Item_Weight'].mean()))
data_train.Outlet_Size = data_train.Outlet_Size.fillna("Small")
data_train.isnull().sum().sum()


# In[12]:


data_test['Item_Weight'] = data_test['Item_Weight'].fillna((data_test['Item_Weight'].mean()))
data_test.Outlet_Size = data_test.Outlet_Size.fillna("Small")
data_test.isnull().sum().sum()


# # **Divide the data in Categorical and Numerial Data**

# In[13]:


column_num = data_train.select_dtypes(exclude = ["object"]).columns
column_object = data_train.select_dtypes(include = ["object"]).columns

test_column_num = data_test.select_dtypes(exclude = ["object"]).columns
test_column_object = data_test.select_dtypes(include = ["object"]).columns


# In[14]:


data_train_num = data_train[column_num]
data_train_object = data_train[column_object]

data_test_num = data_test[test_column_num]
data_test_object = data_test[test_column_object]


# # **Overview Stats About Data**

# In[15]:


data_train_num.describe()


# In[16]:


data_train_object.describe()


# # **Univariate Analysis**

# In[17]:


sns.distplot(data_train_num["Item_Outlet_Sales"]);


# In[18]:


print(("Skewness: %f" % data_train_num["Item_Outlet_Sales"].skew()))
print(("Kurtosis: %f" % data_train_num["Item_Outlet_Sales"].kurt()))


# **Observation** - Target Data is positively skewed

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
data_train_num.hist(figsize=(10,8),bins=6,color='Y')
plt.tight_layout()
plt.show()


# **Observation** -
# *  Most of the MRP is in Range  100 to 180
# *  Around 5000 Item Outlet sale is in between 0 to 2000
# *  Around 2500 items have a weight 10 to 12.5
# *  Most of the Item visibilty is between 0 to 0.05
# 

# In[20]:


plt.figure(1)
plt.subplot(321)
data_train['Outlet_Type'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='green')

plt.subplot(322)
data_train['Item_Fat_Content'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='yellow')

plt.subplot(323)
data_train['Item_Type'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='red')

plt.subplot(324)
data_train['Outlet_Size'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='orange')

plt.subplot(325)
data_train['Outlet_Location_Type'].value_counts(normalize=True).plot(figsize=(10,12),kind='bar',color='black')

plt.subplot(326)
data_train['Outlet_Establishment_Year'].value_counts().plot(figsize=(10,12),kind='bar',color='olive')


plt.tight_layout()
plt.show()


# **Observation** -
# * More than 65 % Outlet  type is  SuperMarket Type 1 
# *  Item Fat Content same category has multiple names i.e "Low fat" as "LF", "low fat" ; "Regular" as reg (require to be merge in single name)
# * Most of Item are from category Fruits and Vegetables and Snacks Foods
# * More than 55 % outlet are of small size
# * More than 40 % Outlet are in tier 3 location followed by Tier2
# * Highest Outlet Opened in year 1985 more than 1400

# **Based on above observation merge "Item Fat Content " Category**

# In[21]:


data_train['Item_Fat_Content'].value_counts()


# In[22]:


data_test['Item_Fat_Content'].value_counts()


# In[23]:


vals_to_replace = {'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'}
data_train['Item_Fat_Content'] = data_train['Item_Fat_Content'].map(vals_to_replace)
data_test['Item_Fat_Content'] = data_test['Item_Fat_Content'].map(vals_to_replace)


# In[24]:


data_train['Item_Fat_Content'].value_counts(normalize=True).plot(figsize=(5,4),kind='bar',color='green')


# **Observation** - after changes we can infer around 75 %  of the items in Mart are Low Fat

# In[25]:


data_train["Outlet_Identifier"].value_counts()


# # **BIVARIATE ANALYSIS**

# In[26]:


ax = sns.catplot(x="Outlet_Identifier", y = "Item_Outlet_Sales", data=data_train, height=5, aspect=2,  kind="bar")


# In[27]:


plt.rcParams['figure.figsize']=(10,4)
ax = sns.boxplot(x="Outlet_Type", y="Item_Outlet_Sales", data=data_train)


# In[28]:


ax = sns.boxplot(x="Outlet_Size", y="Item_Outlet_Sales", data=data_train)


# In[29]:


ax = sns.boxplot(x="Item_Fat_Content", y="Item_Outlet_Sales", data=data_train)


# **Observation** -
# *  Item Outet sale is maximum in Oultlet Identifier OUT027 more than 3500
# *  SuperMarket Type 3 has maximum   Item Outet sale whereas Grocery Store has lowest
# *  Medium size Outlet are having most outlet sale are having potentail outlier to convert it to High size Outlet.
# *  "Low fat" Items are having more Item outlet Sale
# 
# 

# In[30]:


sns.pairplot(data_train[data_train_num.columns])


# In[31]:


sns.heatmap(data_train[data_train_num.columns].corr(),annot=True)



# **Observation** -
# * Item Outlet Sale show mild possitive corelation with Item MRP
# * Item Outlet Sale show negative corelation with Item visibility

# # **PRE PROCESSING DATA**

# **One-hot Encoding** -  Converting the Categorical values to numerical 

# In[32]:


total_object = data_train_object.append(data_test_object)
train_object_lenght = len(data_train_object)
total_cat = pd.get_dummies(total_object, drop_first= True)
data_train_object = total_cat[:train_object_lenght]
data_test_object = total_cat[train_object_lenght:]


# In[33]:


data_train_object.head()


# combining one hot encoded data to numerical data

# In[34]:


df_test = pd.concat([data_test_object,data_test_num],axis=1)
df_train = pd.concat([data_train_object,data_train_num],axis=1)
df_train.head()


# Dividing Data to Dependent and Independent Variable

# In[35]:


train_Y = df_train.iloc[:,-1]
train_X=  df_train.iloc[:,0:-1]


# In[36]:


model = sm.OLS(train_Y, train_X)
results = model.fit()
print((results.summary()))


# # **Scaling Data**

# In[37]:


scaler = StandardScaler()
train_scaler = scaler.fit(train_X)
train_scale = train_scaler.transform(train_X)
train_X = pd.DataFrame(train_scale, columns=train_X.columns)

train_scale = train_scaler.transform(df_test)
df_test = pd.DataFrame(train_scale, columns=df_test.columns)


# In[38]:


model = sm.OLS(train_Y, train_X)
results = model.fit()
print((results.summary()))


# In[39]:


type(results.pvalues.index)


# In[40]:


type(results.pvalues)


# In[41]:


col = [value for value in results.pvalues.index if results.pvalues[value] > -0.001  ]
col


# # **Spliting the Data**

# In[42]:


X1_train, X1_test, Y1_train,Y1_test =  train_test_split(train_X[col],train_Y, random_state=33)
print((X1_train.shape))
print((X1_test.shape))
print((df_test.shape))


# # **Principal component Analysis**

# In[43]:


pca_model = PCA(n_components=0.95)
X1_train = pca_model.fit_transform(X1_train)
X1_test = pca_model.transform(X1_test)
test_X = pca_model.transform(df_test[col])
print((X1_train.shape))
print((X1_test.shape))
print((test_X.shape))


# # **Predictions**

# In[44]:


def modelPredection(model,X1_train,Y1_train,X1_test,Y1_test,test_X) :
    model.fit(X1_train,Y1_train)
    Y1_predict = model.predict(X1_test)
    print(("RMSE : %f"%sqrt(mean_squared_error(Y1_test,Y1_predict))))
    return model.predict(test_X)


# In[45]:


#Linear Regression
linear = linear_model.LinearRegression( fit_intercept=True, n_jobs=None,
         normalize=False);
predict_Y = modelPredection(linear,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[46]:


#RidgeCV
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X1_train,Y1_train)

predict_Y = modelPredection(clf,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[47]:


#Kernel Ridge
RR = KernelRidge(alpha=0.6, kernel='polynomial', degree=3, coef0=2.5)
predict_Y = modelPredection(RR,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[48]:


#Lasso
#lasso = Lasso(alpha =1.1, random_state=1)
lasso = Lasso(alpha =16, random_state=100)
predict_Y = modelPredection(lasso,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[49]:


#Elastic Net 
#elastic_net = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
elastic_net = ElasticNet(alpha=0.8)
predict_Y = modelPredection(elastic_net,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[50]:


#Gradient Boosting
#GBR = GradientBoostingRegressor(n_estimators=30, max_depth=2)
GBR = GradientBoostingRegressor()
predict_Y = modelPredection(GBR,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[51]:


#XGB
model_xgb = xgb.XGBRegressor()
predict_Y = modelPredection(model_xgb,X1_train,Y1_train,X1_test,Y1_test,test_X)


# In[52]:


#light Gradient Boosting
model_lgb = lgb.LGBMRegressor()
predict_Y = modelPredection(model_lgb,X1_train,Y1_train,X1_test,Y1_test,test_X)


# From the prediction you can conclude that XG Boost is gIving better predective model. we can further fine tune the models for increased performance. 
# But this is good to go for beginner model.
# 
# please let me know you feedbacks and comments and boost my motivation by **Upvote.**

# ![](http://www.animatedimages.org/data/media/466/animated-thank-you-image-0078.gif)
