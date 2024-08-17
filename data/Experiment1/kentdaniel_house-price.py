#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


df=pd.read_csv("/kaggle/input/housedata/output.csv")
df


# In[3]:


df.describe().style.background_gradient()


# 
# 
# # distribution of target variable

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
plt.hist(df[df['price']<26590000]["price"], color="b",bins=40)
ax.xaxis.grid(True)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()


# As a general rule of thumb:
# 
# * If skewness is less than -1 or greater than 1, the distribution is highly skewed.
# * If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.
# * If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

# In[5]:


# Skew and kurt
print(("Skewness: %f" % df['price'].skew()))
print(("Kurtosis: %f" % df['price'].kurt()))


# # grouping types of variables

# In[6]:


# explore types of features
print(("total features : "+ str(len(df.columns))))
num_ft=[i for i in df.columns if df[i].dtypes !="O"]
print(("number of numeric variables : %i " %len(num_ft)))
non_num_ft=[i for i in df.columns if i not in num_ft]
print(("number of non-numeric variables : %i " %len(non_num_ft)))
nan_ft=[i for i in df.columns if df[i].isnull().sum()>0]
print(("number of variables with nan : %i " %len(nan_ft)))
nan_num_ft=list(set(num_ft).intersection(nan_ft))
print(("number of numeric variables with null: %i " %len(nan_num_ft)))
nan_non_num_ft=list(set(non_num_ft).intersection(nan_ft))
print(("number of numeric variables with null: %i " %len(nan_non_num_ft)))


# * splitting numeric variables to continuous and discrete and other classes

# In[7]:


num_ft


# In[8]:


# check for numeric variables that is discrete
def show_unique(df,col):
    data=[df[c].nunique() for c in col]
    dummy=pd.DataFrame(columns=col,data=[data])
    return dummy.style.background_gradient("Oranges",axis=1)
show_unique(df,num_ft)


# In[9]:


disc_ft=["floors","waterfront","view","condition","bedrooms"]
time_ft=[i for i in num_ft if "yr" in i]
cont_ft=list(set(num_ft)-set(disc_ft+time_ft))

print(("Discrete variables : "+ str(disc_ft)))
print(("Temporal variables : "+ str(time_ft)))
print(("Continuous variables : "+ str(cont_ft)))


# * checking non-numeric features

# In[10]:


non_num_ft


# In[11]:


df[non_num_ft]


# In[12]:


show_unique(df,non_num_ft)


# # distribution

# In[13]:


def plot_cont(df,con_ft,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[con_ft]),1):
        plt.subplot(len(list(con_ft)), 3, i)
        sns.distplot(df[feature])

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
        plt.ylabel('skewness : %2f'%df[feature].skew(), size=15, labelpad=12.5)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()
plot_cont(df,cont_ft,(12,15))


# In[14]:


plot_cont(df,time_ft,(10,8))


# In[15]:


def plot_disc(df,disc_ft,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[disc_ft]),1):
        plt.subplot(len(list(disc_ft)), 3, i)
        sns.countplot(df[feature])

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()
plot_disc(df,disc_ft,(12,15))


# In[16]:


plt.figure(figsize=(20,10))
sns.countplot(y=df["city"],orient="h")
plt.xticks(rotation=(0))
plt.show()


# # Relationship with target variable

# In[17]:


# config
def show_relationship_cont(df,feature,target,size):
    num_ft=feature
    target=target
    df=df
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    sns.color_palette("husl", 7)
    plt.ticklabel_format(useOffset=False, style='plain')
    # visualising some more outliers in the data values

    for i, feature in enumerate(list(df[num_ft]),1):
        plt.subplot(len(list(num_ft)), 3, i)
        sns.scatterplot(x=feature, y=target, palette='Blues', data=df)

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
        plt.ylabel(target, size=15, labelpad=12.5)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()
show_relationship_cont(df,cont_ft,"price",(12,24))


# In[18]:


show_relationship_cont(df,time_ft,"price",(12,8))


# In[19]:


def show_relationship_cat(df,feature,target):
    num_ft=feature
    target=target
    df=df
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=(12, 30))
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    sns.color_palette("husl", 7)
    plt.ticklabel_format(useOffset=False, style='plain')
    # visualising some more outliers in the data values

    for i, feature in enumerate(list(df[num_ft]),1):
        plt.subplot(len(list(num_ft)), 3, i)
        sns.boxplot(x=feature, y=target,data=df)

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
        plt.ylabel(target, size=15, labelpad=12.5)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()
show_relationship_cat(df,disc_ft,"price")


# In[88]:


plt.figure(figsize=(20,30))
sns.boxplot(y=df["city"],x=df["price_log"])
plt.xticks(rotation=(0))
plt.show()


# In[21]:


df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
df['date'].dtypes


# In[22]:


df=df.sort_values(by="date")


# In[23]:


import matplotlib.pyplot as plt
fig,axis=plt.subplots(1,1,figsize=(18,3))
sns.lineplot(df["date"],df["price"])


# # Outlier removal

# In[24]:


df[df["price"]>5e6]


# In[25]:


# observation: the price above 5e6 is very few
df.drop(df[df["price"]>5e6].index,axis=0,inplace=True)


# In[26]:


# observation: with price 0 is not logical
df.drop(df[df["price"]==0].index,axis=0,inplace=True)


# In[27]:


# observation : country only have 1 unique value it shouldn't affect anything if it is removed
df.drop(["country"],axis=1,inplace=True)


# In[28]:


len(df[df["bedrooms"]==9])


# In[29]:


# observation : only 1 house with 9 bedrooms
df.drop(df[df["bedrooms"]==9].index,axis=0,inplace=True)


# In[40]:


# too complex to extract info
df.drop(columns=["statezip","street"],inplace=True)


# # feature engineering and transformation

# In[36]:


df[cont_ft].corr().style.background_gradient("Oranges")


# In[45]:


from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit_transform(df[["sqft_above","sqft_basement","sqft_living"]])
df["sqft_living_t"]=pca.fit_transform(df[["sqft_above","sqft_basement","sqft_living"]])


# In[55]:


df["sqft_living_t"].skew()


# In[57]:


df.price.skew()


# In[61]:


df["price_log"]=np.log1p(df.price)


# In[65]:


df["price_log"].skew()


# In[67]:


(df["sqft_above"]).skew()


# In[69]:


df["sqft_above_log"]=np.log1p(df["sqft_above"])


# In[71]:


df["sqft_above_log"].skew()


# In[70]:


df[["bathrooms","sqft_living_t","sqft_lot","sqft_basement","sqft_above_log","price_log"]].corr().style.background_gradient("Oranges")


# In[90]:


redundant_cont=["sqft_lot","sqft_above","sqft_living","price"]


# In[74]:


disc_ft


# In[77]:


df[disc_ft+["price"]].corr().style.background_gradient("Oranges")


# In[78]:


time_ft


# In[85]:


df["yr_sold"]=df["date"].dt.year
df["age"]=df["yr_sold"]-df["yr_built"]
df["yr_ren"]=df["yr_sold"]-df["yr_renovated"]


# In[86]:


redundant_year=["yr_built","yr_renovated","date"]


# In[91]:


df.drop(columns=redundant_year+redundant_cont,axis=1,inplace=True)


# In[98]:


df.drop("yr_sold",axis=1,inplace=True)


# In[110]:


df.drop("sqft",axis=1,inplace=True)


# In[99]:


sns.heatmap(df.corr())


# In[100]:


df=pd.get_dummies(df,columns=["city"])


# In[103]:


df.drop_duplicates()


# # feature importance

# In[111]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define dataset
X, y = df.drop("price_log",axis=1),df["price_log"]
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# # summarize feature importance
# for i,v in enumerate(importance):
#       print('Feature: %s, Score: %.5f' % (df.columns[i],v))
# plot feature importance
plt.figure(figsize=(30,30))
plt.barh([df.columns[x] for x in range(len(importance))], importance)
plt.show()


# In[112]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=df.drop("price_log",axis=1)
y=df["price_log"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
m=LinearRegression()
m.fit(X_train,y_train)


# In[142]:


from sklearn.metrics import mean_squared_error,r2_score
mean_squared_error(np.expm1(y_test),np.expm1(m.predict(X_test)))**0.5


# In[143]:


r2_score(np.expm1(y_test),np.expm1(m.predict(X_test)))


# In[132]:


plt.figure(figsize=(40,20))
plt.scatter([i for i in range(len(X_test))],m.predict(X_test),color="red")
plt.scatter([i for i in range(len(X_test))],y_test,color="blue")


# In[146]:


# inverse transform
plt.figure(figsize=(40,20))
plt.plot([i for i in range(len(X_test))],sorted(np.expm1(m.predict(X_test))),color="red")
plt.scatter([i for i in range(len(X_test))],sorted(np.expm1(y_test)),color="blue")


# In[ ]:




