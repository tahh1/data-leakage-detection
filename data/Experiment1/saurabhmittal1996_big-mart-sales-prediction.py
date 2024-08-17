#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#for Spliting Data and Hyperparameter Tuning 
from sklearn.model_selection import train_test_split,GridSearchCV

#Importing Machine Learning Model
from catboost import CatBoostRegressor
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from xgboost import XGBRFRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor

#statistical Tools
from sklearn import metrics

#To tranform data
from sklearn import preprocessing

#Setting Format
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_columns = None
pd.options.display.max_rows = None
np.random.seed(100)


# # Importing Data

# In[2]:


train = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")
test = pd.read_csv("../input/big-mart-sales-prediction/Test.csv")


# In[3]:


data = pd.concat([train,test],ignore_index=True)
data.shape


# In[4]:


print((train.shape,test.shape))
print((train.shape[0]+test.shape[0]))


# In[5]:


data.tail(10)


# # Explorative Data Analysis

# In[6]:


data.info()


# In[7]:


data.describe(include='all').transpose()


# Lets Look at all the unique values in Columns

# In[8]:


#Extracting the unique values of each columns

for i in data.columns:
    print((i," : distinct_value"))
    print((data[i].nunique(), ":No of unique values"))
    print((data[i].unique()))
    print(("-"*30))
    print("")


# # Univariate Analysis

# In[9]:


sns.boxplot(train.Item_Outlet_Sales)


# In[10]:


plt.hist(data.Item_Visibility, bins=10)
plt.show()


# In[11]:


plt.hist(data.Item_MRP, bins=10, rwidth=.8)
plt.show()


# In[12]:


plt.figure(figsize=(10,7))
for i, col in enumerate(['Item_Weight', 'Item_Visibility', 'Item_MRP']):
    plt.subplot(3,1,i+1)
    sns.boxplot(data[col])
    plt.xlabel('')
    plt.ylabel(col)


# In[13]:


plt.figure(figsize=(10,7))
for i, col in enumerate(['Item_Weight', 'Item_Visibility', 'Item_MRP']):
    plt.subplot(3,1,i+1)
    sns.violinplot(data[col])
    plt.xlabel('')
    plt.ylabel(col)


# In[14]:


plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.barplot(data=data, y='Item_Outlet_Sales', x='Item_Fat_Content')
plt.xlabel('Item_Fat_Content', fontsize=14)

plt.subplot(1,2,2)
sns.boxplot(data=data, y='Item_Outlet_Sales', x='Item_Fat_Content')
plt.xlabel('Item_Fat_Content', fontsize=14)
plt.show()


# In[15]:


plt.figure(figsize=(18,8))
sns.countplot(data.Item_Type)
plt.xticks(rotation=20)
plt.show()


# In[16]:


plt.figure(figsize=(18,8))
sns.boxplot(data=data, y='Item_Outlet_Sales', x='Item_Type')
plt.xlabel('Item_Type', fontsize=14)
plt.xticks(rotation=35)
plt.show()


# In[17]:


plt.figure(figsize=(18,8))
sns.countplot(data.Outlet_Identifier)
plt.xticks(rotation=20)
plt.show()


# In[18]:


plt.figure(figsize=(8,5))
sns.countplot(data.Outlet_Size)
plt.xticks(rotation=35)
plt.show()


# In[19]:


plt.figure(figsize=(18,8))
sns.countplot(data.Outlet_Establishment_Year)
plt.xticks(rotation=30)
plt.show()


# In[20]:


plt.figure(figsize=(12,8))
sns.countplot(data.Outlet_Type)
plt.xticks(rotation=45)
plt.show()


# In[21]:


plt.figure(figsize=(25,15))
outlet_col = [ 'Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type','Outlet_Establishment_Year']
for i, col in enumerate(outlet_col):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=data, y='Item_Outlet_Sales', x=col)
    plt.xlabel(col, fontsize=14)


# # Multi-variate Analysis

# In[22]:


plt.figure(figsize=(15,10))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weight and Item_Outlet_Sales Analysis")
sns.scatterplot(data.Item_Weight, data.Item_Outlet_Sales, hue=data.Item_Type)


# In[23]:


plt.figure(figsize=(15,10))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
sns.scatterplot(data.Item_Visibility, data.Item_Outlet_Sales, hue=data.Item_Type)


# # Bivariate Analysis

# In[24]:


plt.figure(figsize=(15,10))
plt.xlabel("Item_Visibility")
plt.ylabel("Market Retail Price")
plt.title("Item_Visibility and Market Retail Price Analysis")
sns.scatterplot(data.Item_Visibility, data.Item_MRP, alpha=0.3)


# In[25]:


sns.scatterplot(data.Item_MRP, data.Item_Outlet_Sales)


# In[26]:


plt.figure(figsize=(18,8))
sns.boxplot(data.Item_Type, data.Item_Outlet_Sales)
plt.xticks(rotation=90)
plt.show()


# In[27]:


plt.figure(figsize=(12,8))
sns.boxplot(data.Item_Fat_Content, data.Item_Outlet_Sales)
plt.xticks(rotation=90)
plt.show()


# In[28]:


plt.figure(figsize=(18,8))
sns.boxplot(data.Outlet_Identifier, data.Item_Outlet_Sales)
plt.xticks(rotation=90)
plt.show()


# In[29]:


sns.violinplot(data.Outlet_Location_Type, data.Item_Outlet_Sales)
plt.xticks(rotation=20)
plt.show()


# In[30]:


plt.figure(figsize=(10,4))
sns.violinplot(data.Outlet_Type, data.Item_Outlet_Sales)
plt.xticks(rotation=20)
plt.show()


# In[31]:


sns.pairplot(data.drop(columns='Outlet_Establishment_Year'), hue='Outlet_Type')


# # Finding Missing Values

# In[32]:


#We can see that we have some data missing

round(100*(data.isna().sum())/len(data), 2)


# In[33]:


#Looking for any '0' Values
data[data==0].sum()


# 1. Filling the mssing data in Item_weight with mean
# 2. Because every item should have some weight
# 

# In[34]:


data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)


# In[35]:


data.head()


# # Correlation

# In[36]:


data.corr()


# In[37]:


#Checking the Correlation between data with respect to target

data.corr()['Item_Outlet_Sales']


# In[38]:


plt.figure(figsize=(25,15))
sns.heatmap(data.corr(), vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Correlation between different attributes")


# In[39]:


data.head()


# # Transform Data

# In[40]:


item_visiblity_avg = data.pivot_table( index = 'Item_Identifier', values = 'Item_Visibility')
item_visiblity_avg.head()


# In[41]:


data['Item_visiblity_avg'] = data.apply(lambda x: x['Item_Visibility']/item_visiblity_avg['Item_Visibility'][item_visiblity_avg.index==x['Item_Identifier']][0], axis=1).astype(float)


# In[42]:


data.head()


# In[43]:


data['Item_Visibility'], _ = stats.boxcox(data['Item_Visibility'] + 1)


# In[44]:


sns.distplot(data['Item_Visibility'])


# In[45]:


'''

We saw that in Item_identifier 1st 2 character's are common and other characters describes about the product

So we can extract those characters to simplify our dataset

'''

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'].value_counts()


# In[46]:


data.head()


# Rearranging the Columns To get the better view of data

# In[47]:


columns = ['Item_Identifier', 'Item_Type_Combined','Item_Type', 'Item_Fat_Content', 'Item_Weight', 'Item_Visibility',
        'Item_visiblity_avg', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales']
data = data[columns]
data.head()


# In[48]:


data.pivot_table(values="Item_Outlet_Sales",index=['Item_Type_Combined','Item_Type'],aggfunc='sum')


# After we extract 1st 2 characters from Item_identifier we saw that it means to certain item type
# Now, we can rename them accordingly

# In[49]:


data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined'].value_counts()


# In[50]:


data.Outlet_Size.value_counts()


# We saw that Outlet_size have some Nan Values Lets fill them with 'Not_specified' for now. We will work later for it.

# In[51]:


data['Outlet_Size'] = data['Outlet_Size'].fillna('Not_specified')
data['Outlet_Size'].value_counts()


# In[52]:


data['Item_Fat_Content'].value_counts()


# After looking closely we get that ["LF"," Low Fat"," low fat"] & ["Regular","reg"] are same and just acronyms

# In[53]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('LF','Low Fat')

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('low fat','Low Fat')

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('reg','Regular')

data['Item_Fat_Content'].value_counts()


# In[54]:


data.pivot_table(values="Item_Outlet_Sales",index=['Outlet_Location_Type','Outlet_Size','Outlet_Type'],aggfunc=np.sum)


# In[55]:


plot_outlet = data.pivot_table(values="Item_Outlet_Sales",index=['Outlet_Location_Type','Outlet_Size','Outlet_Type'],aggfunc=np.sum)
plot_outlet.plot(kind='bar',figsize = (10,6))
plt.xticks(rotation=75)
plt.show()


# In[56]:


plot_item = data.pivot_table(values="Item_Outlet_Sales",index=['Item_Fat_Content','Item_Type_Combined'],aggfunc='sum')
plot_item.plot(kind='bar',figsize = (10,6))
plt.xticks(rotation=0)
plt.show()


# In[57]:


data.pivot_table(values="Item_Outlet_Sales",index=['Item_Fat_Content','Item_Type_Combined','Item_Type'],aggfunc='sum')


# From above table we saw that Non_Consumable goods don't fit with Low_Fat.
# So lets change them accordingly
# Products looks like Household Items and lets it to Household

# In[58]:


data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Household"


# In[59]:


plot_identifier = data.pivot_table(index="Outlet_Identifier", values="Item_Outlet_Sales", aggfunc=np.sum)
plot_identifier


# In[60]:


plot_identifier.plot(kind='bar',figsize = (15,12))
plt.show()


# In[61]:


data.pivot_table(values=['Item_Outlet_Sales'],index=['Outlet_Type','Outlet_Location_Type','Outlet_Size','Outlet_Identifier'],aggfunc=np.sum)


# In[62]:


data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].sum()


# In[63]:


data.Outlet_Identifier.value_counts()


# 1. When we look Closely Grocery Store have the Lowest Sales
# 2. Supermarket Type3 have the Highest sale
# 
# After Carefully analyzing Outlet_size and Outlet_Identifier with Sales and Total Number of each Outlet_Identifier I think some Outlet_Identifier are in wrong Outlet_Size lets fix them

# In[64]:


data.loc[data['Outlet_Identifier']=="OUT010",'Outlet_Size'] = "Small"

data.loc[data['Outlet_Identifier']=="OUT019",'Outlet_Size'] = "Small"

data.loc[data['Outlet_Identifier']=="OUT027",'Outlet_Size'] = "High"

data.loc[data['Outlet_Identifier']=="OUT017",'Outlet_Size'] = "High"

data.loc[data['Outlet_Identifier']=="OUT045",'Outlet_Size'] = "High"

data.loc[data['Outlet_Identifier']=="OUT035",'Outlet_Size'] = "Medium"

data.loc[data['Outlet_Identifier']=="OUT046",'Outlet_Size'] = "Medium"


# In[65]:


data.pivot_table(values=['Item_Outlet_Sales'],index=['Outlet_Size','Outlet_Identifier'],aggfunc=np.sum)


# In[66]:


data.pivot_table(values=['Item_Outlet_Sales'],index=['Outlet_Type','Outlet_Identifier'],aggfunc=np.sum).plot(kind='bar',figsize = (10,6))
plt.show()


# # Let's Create New Columns

# In[67]:


perishable = ["Breads", "Breakfast", "Dairy", "Snack Foods",
               "Fruits and Vegetables", "Meat", "Seafood", "Starchy Foods"]

non_perishable = ["Baking Goods", "Canned", "Frozen Foods", 
                   "Hard Drinks", "Health and Hygiene",
                   "Household", "Soft Drinks"]


# In[68]:


def filter_data(item):
    if item in perishable:
        return 'perishable'
    elif item in non_perishable:
        return 'non_perishable'
    else:
        return'Not_Known'
    
data['Item_Type_New'] = data.Item_Type.apply(filter_data)


# In[69]:


data.Item_Type_New.value_counts()


# In[70]:


sns.scatterplot(data.Item_MRP, data.Item_Outlet_Sales)


# In[71]:


data['Item_MRP_cat'] = pd.cut(data.Item_MRP, bins=[31,69,137,203,270], labels=['a','b','c','d'])


# In[72]:


data.groupby(['Item_Type_Combined', 'Item_Type_New', 'Item_Type'])['Item_Outlet_Sales'].sum()


# In[73]:


data['MRP_per_unit_weight'] = data.Item_MRP/data.Item_Weight    


# In[74]:


data['Outlet_Years'] = 2020 - data['Outlet_Establishment_Year']


# In[75]:


data.Outlet_Establishment_Year = data.Outlet_Establishment_Year.astype('category')


# In[76]:


data['Item_Identifier'] = data['Item_Identifier'].apply(lambda x: x[3:5])


# In[77]:


data.head()


# In[78]:


data.corr()


# In[79]:


data['Item_visiblity_avg'] = np.log(data['Item_visiblity_avg'] + 1)


# In[80]:


plt.figure(figsize=(25,15))
sns.heatmap(data.corr(), vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Correlation between different attributes")


# In[81]:


df = data.copy()


# # Seperating Train and test data

# In[82]:


df_train = df.iloc[0:train.shape[0]]
df_test = df.iloc[train.shape[0]:]


# In[83]:


print((df_train.shape,df_test.shape))


# Now we will do predict our data with get_dummies

# In[84]:


df.info()


# In[85]:


df_train.select_dtypes(include='float').columns


# In[86]:


df_train = pd.get_dummies(df_train)


# In[87]:


df_train.corr()['Item_Outlet_Sales']


# In[88]:


df_train.head()


# In[89]:


df_train.shape


# In[90]:


df_train.info()


# In[91]:


df_x = df_train.drop(['Item_Outlet_Sales'], axis=1)
df_y = df_train['Item_Outlet_Sales']


# In[92]:


x_train, x_test, y_train, y_test=train_test_split(df_x,df_y, train_size=0.8, random_state=10)


# # Now Lets use Machine Learning Model and check which model works

# In[109]:


#pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git


# In[113]:


lr = LinearRegression()

rfc = ensemble.RandomForestRegressor(n_estimators=400, bootstrap=True, min_samples_leaf=100, min_samples_split=8, max_depth=6)
ada = ensemble.AdaBoostRegressor(n_estimators=1000, learning_rate=0.01)
gbr = ensemble.GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000, max_depth=5, min_samples_split=8, min_samples_leaf=100)
xgb = XGBRFRegressor(n_jobs=-1, n_estimators=1000, max_depth=5)
cat = CatBoostRegressor(verbose=0)
dtr = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
lgbr = LGBMRegressor(n_estimators = 440, learning_rate=0.01, max_depth=12, objective='tweedie', num_leaves=15, num_threads = 4)

knn = KNeighborsRegressor()

mlp = MLPRegressor()

svr = SVR(kernel='linear', C=10, gamma='scale')


# In[95]:


accuracy = {}
rmse = {}
explained_variance = {}
max_error = {}
MAE = {}

def train_model(model, model_name):
    print(model_name)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)

    acc = metrics.r2_score(y_test, pred)*100
    accuracy[model_name] = acc
    print(('R2_Score',acc))

    met = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print(('RMSE : ', met))
    rmse[model_name] = met

    var = (metrics.explained_variance_score(y_test, pred))
    print(('Explained_Variance : ', var))
    explained_variance[model_name] = var

    error = (metrics.max_error(y_test, pred))
    print(('Max_Error : ', error))
    max_error[model_name] = error
    
    err = metrics.mean_absolute_error(y_test, pred)
    print(("Mean Absolute Error", err))
    MAE[model_name] = err


# In[96]:


train_model(cat, "Cat Boost")


# In[97]:


train_model(svr, "Support Vector Machine")


# In[98]:


train_model(lr, "Linear Regression")


# In[99]:


train_model(rfc, "Random Forest")


# In[114]:


train_model(xgb, "Xtreme Gradient Random Forest")


# In[101]:


train_model(ada, "Ada Boost")


# In[102]:


train_model(gbr, "Gradient Boost")


# In[103]:


train_model(dtr, "Decision Tree")


# In[104]:


train_model(mlp, "Multi-layer Perceptron")


# In[105]:


train_model(knn, "K Nearest Neighbors")


# In[106]:


train_model(lgbr, 'Light Gradient Boost')


# In[112]:


from ngboost import NGBRegressor
ngb = NGBRegressor(minibatch_frac=0.5, col_sample=0.5, Base=dtr)
train_model(ngb, "Natural Grading Boost")


# # Visualising our Output

# In[115]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,110,5))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(accuracy.keys()), y=list(accuracy.values()), palette="cubehelix")
plt.show()


# In[117]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1600,100))
plt.ylabel("MSE")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(rmse.keys()), y=list(rmse.values()))
plt.show()


# In[118]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1,0.05))
plt.title("Explained Variance Score")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(explained_variance.keys()), y=list(explained_variance.values()))
plt.show()


# In[119]:


sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,10000,500))
plt.ylabel("Max Error")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x=list(max_error.keys()), y=list(max_error.values()))
plt.show()


# In[120]:


sns.set_style("whitegrid")
plt.figure(figsize=(16, 5))
plt.yticks(np.arange(0, 1000, 100))
plt.ylabel("Mean Absolute Error")
plt.xlabel("Algorithms")
plt.xticks(rotation = 90)
sns.barplot(x = list(MAE.keys()), y = list(MAE.values()))
plt.show()


# # Lets Do HyperParameter Tuning

# We will use Gradient Boosting Regressor because we get less error with this method and more accuracy

# In[121]:


'''    
ngb = NGBRegressor(minibatch_frac=0.5, col_sample=0.5, Base=dtr)
param = dict(n_estimators = [np.linspace(0,2000, 1000)])

rs = GridSearchCV(ngb, param, n_jobs=-1, cv=5, verbose=3)

rs.fit(df_x, df_y)

rs.best_score_*100'''

'''
best_estimator = NGBRegressor(minibatch_frac=0.5, col_sample=0.5, Base=dtr, n_estimators=480)'''


# In[122]:


ngb_1 = NGBRegressor(minibatch_frac=0.5, col_sample=0.5, Base=dtr, n_estimators=480)


# In[134]:


train_model(ngb_1,'extra')


# > New Accuracy is a little bit higher "64.89206436566914"
# Lets use it for our New Prediction

# In[123]:


ngb_1.fit(df_x, df_y)


# # Now Let's predict Sales

# In[124]:


df_test.drop(['Item_Outlet_Sales'], axis=1, inplace=True)
df_test.reset_index(drop=True, inplace=True)


# In[125]:


df_test.shape


# In[126]:


df_test_ = pd.get_dummies(df_test)
df_test_.reset_index(drop=True, inplace=True)


# In[127]:


df_test['Item_Outlet_Sales'] = ngb_1.predict(df_test_)


# In[128]:


df_test.head()


# In[129]:


df_test.to_csv('Predicted_sales.csv', index=None)


# In[130]:


submission = ['Item_Type_Combined','Outlet_Identifier','Item_Outlet_Sales']


# In[131]:


submission = df_test[submission]
submission.head()


# In[132]:


submission.to_csv('submission.csv', index=None)


# In[ ]:




