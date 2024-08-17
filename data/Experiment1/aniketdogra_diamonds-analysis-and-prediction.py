#!/usr/bin/env python
# coding: utf-8

# # Diamond Price Modelling
# 
#  **What are diamonds ?**
# 
# > Diamond is a solid form of the element carbon with its atoms arranged in a crystal structure called diamond cubic.
# The most familiar uses of diamonds today are as gemstones used for adornment, and as industrial abrasives for cutting hard materials.
# 
# 
# 
#  **In this notebook, we will try to build a model to predict the prices of diamonds based on various features of diamond  like carat weight, cut quality ,etc.**
#  
# *Dataset used in this notebook has been taken from [KAGGLE](https://www.kaggle.com/shivam2503/diamonds)*

# ## TOPICS
# 
# 1. [**A Quick Look at the Dataset**](#link1)
# 2. [**Exploring Correlation between Features**](#link2)
# 3. [**Splitting Data into Test and Train Set**](#link3)
# 4. [**Data Visualisation**](#link4)
# 5. [**Preparing Data for ML algorithm**](#link5)
# 6. [**Applying ML Algorithm on the Dataset**](#link6)
# 7. [**Conclusion**](#link7)

# <a id="link1"></a>
# ## A Quick Look at the Dataset

# ### Importing the important libraries required for this project and getting the data from the dataset
# 

# In[1]:


import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings(action = "ignore")

get_ipython().run_line_magic('matplotlib', 'inline')
diamonds = pd.read_csv("../input/diamonds.csv")


# **Now let's take a look at our diamond dataset.**

# In[2]:


diamonds.head()


# In[3]:


diamonds.info()


# ### Features of the Dataset
# 
# - **Carat** weight of the diamond
# - **cut** Describe cut quality of the diamond. Quality in increasing order Fair, Good, Very Good, Premium, Ideal - - - **color** Color of the diamond, with D being the best and J the worst
# - **clarity** How obvious inclusions are within the diamond:(in order from best to worst, FL = flawless, I3= level 3 inclusions) FL,IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
# - **depth** The height of a diamond, measured from the culet to the table, divided by its average girdle diameter
# - **table** The width of the diamond's table expressed as a percentage of its average diameter
# - **price** the price of the diamond
# - **x** length mm
# - **y** width mm
# - **z** depth mm

# In[4]:


diamonds["cut"].value_counts()


# In[5]:


diamonds["color"].value_counts()


# In[6]:


diamonds["clarity"].value_counts()


# ### Dropping the unnecessary column Unnamed: 0

# In[7]:


# Price is of different data type and unnecessary column "Unnamed"
diamonds = diamonds.drop("Unnamed: 0",axis = 1)
diamonds["price"] = diamonds["price"].astype("float64")


# In[8]:


diamonds.head()


# In[9]:


diamonds.describe()


# ### Plotting Histogram to get an idea about the different features/attributes of the dataset

# In[10]:


diamonds.hist(bins = 50, figsize = (20,15))
plt.show()


# <a id="link2"></a>
# ## Exploring Correlation between Features

# In[11]:


corr_matrix = diamonds.corr()

plt.subplots(figsize = (10,8))
sns.heatmap(corr_matrix, annot = True, cmap = "Blues")
plt.show()


# ### Conclusions
#  - **x , y and z are correlated with the price.** 
#  - **Price of the diamond and carat weight of the diamond are highly correlated**
#  - **Depth and Table are weakly correlated with the price of the diamond.**
#  - **Carat is one of the main features to predict the price of a diamond.**

# <a id="link3"></a>
# ### Splitting Data into Test and Train Set
# 
# It is advisable to split the dataset into Test set (80%) and Train set (20%). The test set allows our model to make 
# predictions on values which it has never seen before.
# 
# But taking random samples from our dataset can introduce significant **sampling bias**. Therefore, in order to avoid sampling bias, the data will be divide into different homogenous subgroups called strata. This is called **Stratified Sampling**. Since, we know that carat is the most important parameter to predict the price of the diamonds we will use it for Stratified sampling 

# In[12]:


diamonds["carat"].hist(bins = 50)
plt.show()


# In[13]:


diamonds["carat"].max()


# In[14]:


diamonds["carat"].min()


# Most of the carat value ranges from 0.3 to 1.2. So, we will divide the carat into 5 categories.

# In[15]:


# Divide by 0.4 to limit the number of carat strata

diamonds["carat_cat"] = np.ceil(diamonds["carat"]/0.4)

# Label those above 5 as 5
diamonds["carat_cat"].where(diamonds["carat_cat"] < 5, 5.0, inplace = True)


# In[16]:


diamonds["carat_cat"].value_counts()


# In[17]:


diamonds["carat_cat"].hist()


# Now we will perform the stratified splitting of the dataset using sklearn's StratifiedShuffleSplit class

# In[18]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(diamonds,diamonds["carat_cat"]):
    strat_train_set = diamonds.loc[train_index]
    strat_test_set = diamonds.loc[test_index]
    


# In[19]:


strat_test_set["carat_cat"].value_counts() / len(strat_test_set)


# We will now drop the carat category columns.

# In[20]:


for x in (strat_test_set, strat_train_set):
    x.drop("carat_cat", axis=1,inplace = True)


# In[21]:


strat_test_set.describe()


# Size of Test Set = 10788

# In[22]:


strat_train_set.describe()


# Size of Train Set = 43152

# <a id="link4"></a>
# ## Data Visualisation 
# 
# We will be using training set to plot varoius graphs to visualise and draw conclusions from the data.

# In[23]:


diamonds = strat_train_set.copy()


# ### Plotting scatterplot between price and carat

# In[24]:


diamonds.plot(kind="scatter", x="price", y="carat",alpha = 0.1)
plt.show()


# ### Count plots of different categorical features of diamonds

# In[25]:


fig, ax = plt.subplots(3, figsize = (14,18))
sns.countplot('cut',data = diamonds, ax=ax[0],palette="Spectral")
sns.countplot('clarity',data = diamonds, ax=ax[1],palette="deep")
sns.countplot('color',data = diamonds, ax=ax[2],palette="colorblind")
ax[0].set_title("Diamond cut")
ax[1].set_title("Diamond Clarity")
ax[2].set_title("Diamond Color")
plt.show()


# ### Comparison of carat with price based on diamond cut.

# In[26]:


sns.pairplot(diamonds[["price","carat","cut"]], markers = ["o","v","s","p","d"],hue="cut", height=5)
plt.show()

f, ax = plt.subplots(2,figsize = (12,10))
sns.barplot(x="cut",y="price",data = diamonds,ax=ax[0])
sns.barplot(x="cut",y="carat",data = diamonds, ax=ax[1])
ax[0].set_title("Cut vs Price")
ax[1].set_title("Cut vs Carat")
plt.show()


# **Conclusion**
# - Fair cut diamonds weigh the most but are not the most expensive diamonds.
# - Premium cut diamonds are the most expensive diamonds.
# - Ideal cut diamonds weigh less and are cheapest diamonds.
# 
# We can see that price of diamond is dependent on the cut.

# ### Comparison of carat with price based on diamond color

# In[27]:


sns.pairplot(diamonds[["price","carat","color"]], hue="color", height=5, palette="husl")
plt.show()

f, ax = plt.subplots(2,figsize = (12,10))
sns.barplot(x="color",y="price",data = diamonds,ax=ax[0])
sns.barplot(x="color",y="carat",data = diamonds, ax=ax[1])
ax[0].set_title("Color vs Price")
ax[1].set_title("Color vs Carat")
plt.show()


# **Conclusions**
# - J color diamonds are the most expensive and the heaviest diamonds.
# - The two plots are very similar.
# 
# Thus, it can be concluded that the heavier diamond is expensive, if only color is considered.

# ### Comparison of carat with price based on diamond clarity

# In[28]:


sns.pairplot(diamonds[["price","carat","clarity"]],hue="clarity", height=5)
plt.show()

f, ax = plt.subplots(2,figsize = (12,10))
sns.barplot(x="clarity",y="price",data = diamonds,ax=ax[0])
sns.barplot(x="clarity",y="carat",data = diamonds, ax=ax[1])
ax[0].set_title("Clarity vs Price")
ax[1].set_title("Clarity vs Carat")
plt.show()


# ### More plots to understand the realtion between cut,color and clarity with prices

# In[29]:


fig, ax = plt.subplots(3, figsize = (14,18))
sns.violinplot(x='cut',y='price',data = diamonds, ax=ax[0],palette="Spectral")
sns.violinplot(x='clarity',y='price',data = diamonds, ax=ax[1],palette="deep")
sns.violinplot(x='color',y='price',data = diamonds, ax=ax[2],palette="colorblind")
ax[0].set_title("Cut vs Price")
ax[1].set_title("Clarity vs Price")
ax[2].set_title("Color vs Price ")
plt.show()


# In[30]:


from pandas.plotting import scatter_matrix

attributes = ["depth","table","x","y","z","price"]
scatter_matrix(diamonds[attributes], figsize=(12, 8))


# <a id="link5"></a>
# ## Preparing data for the ML Algorithms
# 

# In[31]:


sample_incomplete_rows = diamonds[diamonds.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[32]:


diamonds = strat_train_set.drop("price", axis=1)
diamonds_label = strat_train_set["price"].copy()
diamonds_only_num = diamonds.drop(["cut","clarity","color"],axis=1)

diamonds_only_num.head()


# 
# ### Feature Scaling
# 
# 
# Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales. Therefore, it is necessary to feature scale all the features of diamond dataset. There are two ways of doing feature scaling -min-max scaling and standardization. I will be using standardization as it is not affected by any outliers.

# In[33]:


from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
diamonds_scaled_num = std_scaler.fit_transform(diamonds_only_num)

diamonds_scaled_num


# In[34]:


pd.DataFrame(diamonds_scaled_num).head()


# ### Encoding Categorical Attributes
# 
# In this dataset, we have three categorical attributes.ML algorithms work better with numbers.Thus, we will convert them into numbers using OneHotEncoder of scikit learn.

# In[35]:


diamonds_cat = diamonds[["cut","color","clarity"]]
diamonds_cat.head()


# In[36]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
diamonds_cat_encoded = cat_encoder.fit_transform(diamonds_cat)

diamonds_cat_encoded.toarray()


# In[37]:


cat_encoder.categories_


# ### Transformation Pipeline
# 
# We have to perform feature scaling and label encoding on dataset before feeding it into ML algorithms. So, to simplify the process we will create a pipeline using ColumnTransformer which successively performs feature scaling and Label encoding.  

# In[38]:


from sklearn.compose import ColumnTransformer

num_attribs = list(diamonds_only_num)
cat_attribs = ["cut","color","clarity"]
pipeline = ColumnTransformer([
    ("num", StandardScaler(),num_attribs),
    ("cat",OneHotEncoder(),cat_attribs),
])

diamonds_prepared = pipeline.fit_transform(diamonds)


# In[39]:


diamonds_prepared


# In[40]:


pd.DataFrame(diamonds_prepared).head()


# In[41]:


diamonds_prepared.shape


# <a id="link6"></a>
# ## Applying ML Algorithms on the Dataset
# 

# Now, it is time to select a model, train it and evaluate its performance using test set.
# First of all we will import mean_squared_error and cross_val_score from sklearn to evaluate the models.
# 
# We will create one function that will run through each algorithm. We'll also have variables that hold results of the algorithms for future comparisons. RMSE and CV_scores are used to check the performance. The function will plot a graph to show how well our algorithm has predicted the data.

# In[42]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from random import randint

X_test = strat_test_set.drop("price",axis=1)
y_test = strat_test_set["price"].copy()

model_name = []
rmse_train_scores = []
cv_rmse_scores = []
accuracy_models = []
rmse_test_scores = []

def model_performance(modelname,model,diamonds = diamonds_prepared, diamonds_labels = diamonds_label,
                      X_test = X_test,y_test = y_test,
                      pipeline=pipeline, cv = True):
    
    model_name.append(modelname)
    
    model.fit(diamonds,diamonds_labels)
    
    predictions = model.predict(diamonds)
    mse_train_score = mean_squared_error(diamonds_labels, predictions)
    rmse_train_score = np.sqrt(mse_train_score)
    cv_rmse = np.sqrt(-cross_val_score(model,diamonds,diamonds_labels,
                                       scoring = "neg_mean_squared_error",cv=10))
    cv_rmse_mean = cv_rmse.mean()
    
    print(("RMSE_Train: %.4f" %rmse_train_score))
    rmse_train_scores.append(rmse_train_score)
    print(("CV_RMSE: %.4f" %cv_rmse_mean))
    cv_rmse_scores.append(cv_rmse_mean)
    
    
    print("---------------------TEST-------------------")
    
    X_test_prepared = pipeline.transform(X_test)
    
    test_predictions = model.predict(X_test_prepared)
    mse_score = mean_squared_error(y_test,test_predictions)
    rmse_score = np.sqrt(mse_score)
    
    print(("RMSE_Test: %.4f" %rmse_score))
    rmse_test_scores.append(rmse_score)
    
    accuracy = (model.score(X_test_prepared,y_test)*100)
    print(("accuracy: "+ str(accuracy) + "%"))
    accuracy_models.append(accuracy)
    
    start = randint(1, len(y_test))
    some_data = X_test.iloc[start:start + 5]
    some_labels = y_test.iloc[start:start + 5]
    some_data_prepared = pipeline.transform(some_data)
    print(("Predictions:", model.predict(some_data_prepared)))
    print(("Labels:    :", list(some_labels)))
    
    
    plt.scatter(y_test,test_predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.plot(x_lim, y_lim, "go--")
    plt.show()
    
    


# **Linear Regression**

# In[43]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
model_performance("Linear Regression",lin_reg)


# **Decision Tree Regression**

# In[44]:


from sklearn.tree import DecisionTreeRegressor

dec_tree = DecisionTreeRegressor(random_state=42)
model_performance("Decision Tree Regression",dec_tree)


# **Random Forest Regression**

# In[45]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators = 10, random_state = 42)
model_performance("Random Forest Regression",forest_reg)


# **Ridge Regression**

# In[46]:


from sklearn.linear_model import Ridge

ridge_reg = Ridge(normalize = True)
model_performance("Ridge Regression",ridge_reg)


# **Lasso Regression**

# In[47]:


from sklearn.linear_model import Lasso

lasso_reg = Lasso(normalize = True)
model_performance("Lasso Regression",lasso_reg)


# **Elastic Net Regression**

# In[48]:


from sklearn.linear_model import ElasticNet

net_reg = ElasticNet()
model_performance("Elastic Net Regression",net_reg)


# **Ada Boost Regression**

# In[49]:


from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(n_estimators = 100)
model_performance("Ada Boost Regression",ada_reg)


# **Gradient Boosting Regression**

# In[50]:


from sklearn.ensemble import GradientBoostingRegressor

grad_reg = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1,
                                     max_depth = 1, random_state = 42, loss = 'ls')
model_performance("Gradient Boosting Regression",grad_reg)


# ### Comparing the Accuracies of different Regression Models

# In[51]:


compare_models = pd.DataFrame({"Algorithms" : model_name, "Models RMSE" : rmse_test_scores, 
                               "CV RMSE Mean" : cv_rmse_scores, "Accuracy" : accuracy_models})
compare_models.sort_values(by = "Accuracy", ascending=False)


# In[52]:


sns.pointplot("Accuracy","Algorithms",data=pd.DataFrame({'Algorithms':model_name,"Accuracy":accuracy_models}))


# <a id="link7"></a>
# ## Conclusion
# 
# **Random Forest Regressor gives us the Highest accuracy.**
# 
# **THANK YOU**

# In[53]:




