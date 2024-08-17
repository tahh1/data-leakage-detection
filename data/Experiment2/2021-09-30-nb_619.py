#!/usr/bin/env python
# coding: utf-8

# <span style='font-size: 2.5em'><b>São Paulo Housing 🏡</b></span><br/>
# 
# 
# <span style='font-size: 1.5em'>Predicting housing prices in São Paulo's apartments for sale</span>
# 
# <span style="background-color: #ffc351; padding: 4px; font-size: 1em;"><b>Final Sprint </b></span>
# 
# 
# 
# ### **D2APR: Aprendizado de Máquina e Reconhecimento de Padrões** (IFSP, Campinas) <br/>
# **Prof**: Samuel Martins (Samuka) <br/>
# 
# #### Study Project
# 
# **Students**: Carlos Danilo Tomé e Lucas Galdino de Camargo
# 
# **Dataset**: https://www.kaggle.com/argonalyst/sao-paulo-real-estate-sale-rent-april-2019
# 
# This data has about 13,000 apartments in São Paulo City, Brazil, available in Kaggle platform.
# 
# **Final Goal**: Predict housing prices for sale in São Paulo using machile learning models and techniques.
# 
# ---
# 
# ## 🎯 Notebook Goals
# 
# - 0. Imports, settings and data reading
# - 1. Framing the problem
# - 2. EDA and Data Cleanning
# - 3. Modelling
# - 4. Results
# 
# ---

# ### 0. Imports, settings and read data
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


# In[2]:


sns.set_theme(style="whitegrid")

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

import warnings
warnings.filterwarnings("ignore")


path='./datasets/sao-paulo-properties-april-2019.csv'
housing = pd.read_csv(path)

housing.head(3)


# ## 🔲 1. Framing the Problem

# ### 📄 1.1. Context
# 
# **São Paulo** is the most populous city in Brazil. It's population in 2020 is around 12.4 million, acordding to IBGE [1]. 
# 
# It's also the largest economy in Brazil, which PIB in 2018 was R$ 2.210 Bi [2].
# 
# As any large economy, São Paulo has the real state market an important sector of it's economy, making possible the growth of digital companies in the sector as Airbnb and Quinto Andar, for example.
# 
# Given the importance of the sector, we'll challenge ourselves to make and analytical application in order to predict house prices for sale in São Paulo. To do so, we are making use of machine learning models and techniques, as we are learning and studying throughout our post-graduation at IFSP-SP.
# 
# 
# **References:** <br/>
# [1] https://cidades.ibge.gov.br/brasil/sp/sao-paulo/panorama
# 
# [2] https://www.ibge.gov.br/explica/pib.php

# ### 🧠 1.2. Challenge
# Seu Barriga Housing is a well-established real estate company working on the housing market in São Paulo. 
# 
# They are facing a competitive market and they don't want to stay behing. There are several challenges they are facing, as we'll list below:
#    - They have only **10 analysts**: Their capacity is to attend around 500 clients a month;
#    - **Response time**: As the prices are set by analysts, they usually take at leats 1 or 2 days to respond each client;
#    - **Follow up in negotiations**: As the analysts have many clients to attend, and they spend too much time by setting prices on houses, they aren't abble to attend the clients at its bests needs;
#    - **Scalability**: As their current business is structured, they aren't able to scale. The expansion of people wouldn't solve their problems, but would increase considerably their payroll amount;
#    - **NPS**: As a result of Seu Barriga Housing problems to attend their client as they really need, they have suffered with their Net Promoter Score, a metric which measures the satisfaction level of clients. Their current NPS is around 60 (in a scale from 1 to 100). The main compaints Seu Barriga are facing were about client experience and the delivery time of services;
#     
# 
# In order to deliver a better solution to its customers, they are willing to develop a machine learning application to predict house prices, pursuing faster deliveries as well as more precise price estimations. 
# 
# #### 🎯 **Objective:**
# **Build a machine learning solution to automatically predict the housing prices for _apartments_ in sale in São Paulo.** <br/>
# These predictions will be used in Seu Barriga's new application service attending a better service to its customers.
# 
# #### **Baseline:**
# Currently, the **housing prices** are estimated ***manually by experts***: a team gathers up-to-date information about an apartment and finds out the _housing price_. 
# This is _costly_ and _time-consuming_, and their **estimates are not that great**; they often realize that **their estimates were off by more than 20%**.
# 
# #### **Solution Planning:**
# - **Regression problem**
# - Metrics:
#     - R²
#     - Root Mean Squared Error (RMSE)
# - Data sources:
#     - [São Paulo Real Estate Sale](https://www.kaggle.com/argonalyst/sao-paulo-real-estate-sale-rent-april-2019)
# - No assumptions were made
# - Project deliverable:
#     - A simple exploratory data analysis
#     - **A ML system/model** launched in _production_ <br/><br/>

# In[3]:


print(('This dataset have', housing.shape[1],'columns and', housing.shape[0],
              'row about housing in São Paulo distributed in', len(set(housing.District)), 'districts.\n'))
housing.info()


# ### 1.3 Data Structure
# 
# Each row corresponds to an apartment with **15 attributes**  and it's price. There are 8 numeric attributes  and 7 categorical, as described below: <br/>
# 
# 
# * **Price:** Final price advertised (RS Brazilian Real). Column Type: Int64.
# 
# * **Condo:** Condominium expenses (unknown values are marked as zero). Column Type: Int64.     
# 
# * **Size:**  The property size in Square Meters m² (private areas only). Column Type: Int64. 
# 
# * **Rooms:** Number of bedrooms. Column Type: Int64.   
# 
# * **Toilets:** Number of toilets (all toilets). Column Type: Int64.     
# 
# * **Suites:** Number of bedrooms with a private bathroom (en suite). Column Type: Int64.  
# 
# * **Parking:** Number of parking spots. Column Type: Int64.       
# 
# * **Elevator:** Binary value: 1 if there is elevator in the building, 0 otherwise. Column Type: Int64. 
# 
# * **Furnished:** Binary value: 1 if the property is funished, 0 otherwise. Column Type: Int64.   
# 
# * **Swimming Pool:** Binary value: 1 if the property has swimming pool, 0 otherwise. Column Type: Int64.    
# 
# * **New:** Binary value: 1 if the property is very recent, 0 otherwise. Column Type: Int64.  
# 
# * **District:**  The neighborhood and city where the property is located, e.i: Itaim Bibi/São Paulo. 
# 
# * **Negotiation Type:**  Type of negotiation of housing. Column Type: String.
#      * rent
#      * sale 
# 
# * **Property Type:** Type of housing, in this feature we only have one kind of housing: 'apartment'. Column Type: String.   
# 
# * **Latitude:**  Geographic location. Column Type: Geocode.         
# 
# * **Longitude:** Geographic location. Column Type: Geocode. 

# In[4]:


housing.info()


# In[5]:


housing.describe()


# ## 🧹 2. EDA and Data Cleanning

# ### 2.1 Checking for duplicated samples

# In[6]:


housing[housing.duplicated()].head(5)


# **There are 319 duplicated data**, so we will drop duplicated data by keeping the first sample in the dataset.

# ### 2.2 Checking Negotiation Type

# In[7]:


fig, axes = plt.subplots(2, 2, figsize=(14,12))

sns.histplot(x="Price", data=housing[housing['Negotiation Type'] == 'rent'],  ax=axes[0, 0], color = 'tab:blue')
axes[0, 0].set_title('Histogram - Housing Price of rent advertising')

sns.histplot(x="Price", data=housing[housing['Negotiation Type'] == 'sale'],  ax=axes[0, 1], color = 'tab:orange')
axes[0, 1].set_title('Histogram - Housing Price of sale advertising')

sns.boxplot( x="Price", data=housing[housing['Negotiation Type'] == 'rent'],  ax=axes[1, 0], color = 'tab:blue')
axes[1, 0].set_title('Boxplot - Housing Price of rent advertising')

sns.boxplot( x="Price", data=housing[housing['Negotiation Type'] == 'sale'],  ax=axes[1, 1], color = 'tab:orange')
axes[1, 1].set_title('Boxplot - Housing Price of sale advertising');


# In[8]:


##### HUE of negotion type
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.scatterplot(y="Price",x='Condo', data=housing, hue ='Negotiation Type',  ax=axes[0], color = 'tab:blue')
axes[0].set_title('Histogram - Housing Price of rent advertising')

sns.scatterplot(y="Price",x='Size', data=housing,hue ='Negotiation Type',  ax=axes[1], color = 'tab:orange')
axes[1].set_title('Histogram - Housing Price of rent advertising');


# ### 2.3 Dropping Property Type

# We are dropping the column "Property Type" because all instances have the same value in this feature: 'apartment'.

# In[9]:


set(housing['Property Type'])


# ### Applying Data Cleanning

# In[10]:


# Set only sale Negotiation Type
housing = housing[housing['Negotiation Type']=='sale']

# Drop duplicated data
housing.drop_duplicates(keep='first', inplace=True)

# Drop columns
housing.drop(['Negotiation Type', 'Property Type'], axis=1, inplace=True)


# ### 2.4 Geolocation with inconsistent values
# 
# São Paulo's coordinates are [3]:
# 
#     -Latitude: -23.5489, 
#     -Longitude: -46.6388, 
# 
# Let's check on how many records are distant from a range centered at this point.
# 
# [3] https://pt.db-city.com/Brasil--S%C3%A3o-Paulo--S%C3%A3o-Paulo

# In[11]:


min_y= -23.8
max_y= -23.2
min_x= -46.95
max_x= -46

housing[ (housing['Latitude'] < min_y )   |
         (housing['Longitude'] < min_x )  |
         (housing['Latitude']  > max_y )  |
         (housing['Longitude'] > max_x )  ].shape


# Ok, so we have 468 records that are outliers.
# 
# What we are going to do is to replace this outliers by null, and later we'll fill this fields with the meaan latitude and longitude of its district.

# In[12]:


# Replace outlier coodinates as NaN

housing['Latitude'][(housing['Latitude'] < min_y )  |
                   (housing['Longitude'] < min_x )  |
                   (housing['Latitude']  > max_y )  |
                   (housing['Longitude'] > max_x )  
                   ]= np.nan

housing['Longitude'][(housing['Latitude'] < min_y ) |
                   (housing['Longitude'] < min_x )  |
                   (housing['Latitude']  > max_y )  |
                   (housing['Longitude'] > max_x )  
                   ]= np.nan


# In[13]:


# Plot view
housing.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.1, figsize=(10, 7));


# ### 2.5 District Information
# 
# As we've seen before, there are some data with missing or outlier values on Latitude and Longitude fields.
# To deal with that we are first replacing these for nan, and then we will use the mean coordinates of it's district to replace the null cases.
# 
# Another thing we will do is about the districts, because there there are many of them, and we don't want to face any problem with dimensionality, but we don't want to miss information about the district of a given apartment as well.
# What we are going to do is to divide all of São Paulo's districts into 4 hierarchical classes based on the mean price of the square meter in the districts, and use this variable insted of the district itself.

# In[14]:


# String Replace
# Let's drop '/São Paulo' from the District column of all records in the dataframe
housing['District'] = housing['District'].str.partition('/', expand=True)[0]


# In[15]:


district = housing.groupby(['District']).apply(lambda x: pd.Series(dict(
                                                        qtd_housing          = ((x.Price.count()))
                                                        ,mean_housing_price   =  ((x.Price.mean())) 
                                                        ,mean_housing_size    =  ((x.Size.mean()))
                                                        ,mean_Price_per_square_meter =  ((x.Price.sum()))/((x.Size.sum()))
                                                        ,mean_housing_condo   =  ((x.Condo.mean()))
                                                        ,pct_new_housing      = ((x.New.sum())*100/(x.Price.count()))
                                                        ,Latitude_district    =  ((x.Latitude.mean())) 
                                                        ,Longitude_district   =  ((x.Longitude.mean()))
))).reset_index() 

district.describe()


# In[16]:


# Let's take a look at the 15 districts with the highest price per square meter in São Paulo
district.reset_index(drop = True, inplace=False).sort_values(['mean_Price_per_square_meter'], ascending = [False])[:15].\
                            style.background_gradient(cmap='Oranges', axis =0)


# In[17]:


# Plot view
sns.scatterplot(data = district , x="Longitude_district", y="Latitude_district",  hue='mean_Price_per_square_meter');


# In[18]:


# Creating a price rate considering the mean price per square meter into 4 labels with hierarchical values
district['district_rate'] = pd.qcut(district['mean_Price_per_square_meter'], q=4,   labels=[1, 2, 3, 4])


# In[19]:


sns.histplot(x="mean_Price_per_square_meter", data=district, color = 'tab:blue', bins =50,hue="district_rate", multiple="stack");


# In[20]:


sns.scatterplot(data = district , x="Longitude_district", y="Latitude_district",  hue='district_rate');


# #### Saving District information

# In[22]:


district.to_csv('./datasets/district_information.csv', sep=';',index=False)


# ### 2.6 Integer variables

# In[23]:


##### Integer variables - Count values
housing.hist(column=['Elevator', 'Furnished', 'Swimming Pool', 'New','Rooms', 'Toilets', 'Suites', 'Parking']
             , figsize=(12,12), layout = (4,2));


# In[24]:


##### Integer Variabels - Boxplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

sns.boxplot(y="Price",x= 'Rooms',data=housing,  ax=axes[0, 0], color = 'tab:blue')
axes[0, 0].set_title("Housing Sale's Price distribution by the number of rooms")

sns.boxplot(y="Price",x= 'Toilets', data=housing,  ax=axes[0, 1], color = 'tab:orange')
axes[0, 1].set_title("Housing Sale's Price distribution by the number of toilets")

sns.boxplot( y="Price",x= 'Suites',data=housing,  ax=axes[1, 0], color = 'tab:blue')
axes[1, 0].set_title("Housing Sale's distribution by the number of suites")

sns.boxplot( y="Price",x= 'Parking', data=housing,  ax=axes[1, 1], color = 'tab:orange')
axes[1, 1].set_title("Housing Sale's distribution by the number of parking spots");


# In[25]:


# Let's Create some interesting variables to use when modelling
housing_temp = housing.copy()

housing_temp['Total Rooms']            = housing['Rooms'] + housing['Toilets'] + housing['Suites']
housing_temp['Total Bedrooms']         = housing['Rooms'] + housing['Suites']

housing_temp.hist(column=['Condo','Total Rooms', 'Total Bedrooms'], figsize=(12,4),layout=(1,3));


# In[26]:


##### Correlation of the variables with the Sale's Price
housing.corr()[['Price']].sort_values(by=['Price'] ,ascending = False)[1:].style.background_gradient(cmap='Greens', axis =0)


# ### 2.7 Splitting the data into train and test sets 
# 
# Well, at this point we've done pretty much of the basics data treatment we need, created some new variables to use in modelling as well as we've known this dataset a little better. For now we'll divide our dataset into train and test, by stratifying the samples considering the District, so we don't miss any District when training.

# In[27]:


# Splitting the data between train and test
housing_train, housing_test = train_test_split(housing, test_size=0.2, stratify=housing['District'], random_state=42)


# In[28]:


# Saving the data after split
housing_train.to_csv('./datasets/housing_train.csv', sep=';',index=False)
housing_test.to_csv('./datasets/housing_test.csv', sep=';',index=False)


# ### 2.8 Checking samples with missing samples     

# In[29]:


housing.isnull().sum()


# ## 🤖 3. Modelling
# 
# From now one things start to get funnier, its time to model!
# 
# To solve our regression problem we've tried the following models:
# 
#     
# **Random Forest Regressor**: In order to find best parameters we've set 90 total fits in the Grid Search, changing the parameters :  min_samples leaf, min_samples_split and n_estimators.
#             
# **Support Vector Regressor**: In this case we also have set 90 different sets in the Grid Search, by changing the parameters: kernel, degree, epsilon and C.
#             
# **Ridge Linear Model**: In this case, as this model is really simple and fast to run, we've set 400 total fits in the Grid Search, by changing the parameters: alpha, solver, fit_intercept and normalize.
#             
# **Lasso Linear Model**: For this one, we've set 80 total fits in the Grid Search, by changing the parameters: alpha, fit_intercept and normalize.
#             
# **Gradient Boost Regressor**: In this case, we've set 480 total fits for Grid Search, by changing the parameters: subsample, n_estimators, learning_rate and criterion.
#             
# **K Neighbors Regressor**: For this model, we've set 90 total fits for Grid Search, when changing only the parameters: n_neighbors, weights and p.
#     
# **Bayesian Ridge**: In this particular case we've set only 15 different fits for Grid Search, only changing the n_iter parameter.
#             
# **Ada Boost Regressor**: For this model, we've set 120 total fits for Grid Search, by changing the parameters: base_estimator, n_estimators, learning_rate and loss.
#     
# 
# At the end, we've trained over this project over than **1300** different versions of machine learning models!
# 
# For your sake, in this particular notebook we'll keep just 2 models: Random Forest Regressor and Ada Boost Regressor, our champion models.
# 
# If you want to see more about any other model that we've tried, please take a look at the sprint 3 of this project.

# In[3]:


# Importing our Train set and splitting it into features (X_train) and target (y_train)

# Loading trainning data
housing_train = pd.read_csv('./datasets/housing_train.csv', sep=';')

X_train = housing_train.drop(['Price'], axis=1).copy()
y_train = housing_train['Price'].copy()


# Importing Test set and split it into features (X_train) and target (y_train)

# Loading testing data
housing_test = pd.read_csv('./datasets/housing_test.csv', sep=';')

X_test = housing_test.drop(['Price'], axis=1).copy()
y_test = housing_test['Price'].copy()


# ### 3.1 Creating a Column Transformer
# 
# In order to build a full pipeline to our models, we'll create a class to deal with all the data preprocessing and transforming we'll need.

# In[4]:


class pre_processing_transform(BaseEstimator, TransformerMixin):
    
    # class that creates the new variables
    # class that deals with the outliers in the coordinates
    # class that substitutes the District by its created hierarchical class considering its mean price per square meter
    # Goal: customized transformator to consider in the pipeline as well as adapted to sklearn
    # notes: with this configuration this class is able to do "fit", "transform" and "fit_transform", natives on sklearn
    
    def __init__(self):
        return None
    
    def fit(self,X,y = None):
        return self
    
    def transform(self, X, y = None):
        
        temp = X.copy()
        
        # Create new features
        temp['Total Rooms']            = temp['Rooms'] + temp['Toilets'] + temp['Suites']
        temp['Total Bedrooms']         = temp['Rooms'] + temp['Suites']
        
        # Add District Rate
        district = pd.read_csv('./datasets/district_information.csv', sep=';')
        
        temp = pd.merge(temp, district[['District','district_rate','Latitude_district','Longitude_district']],
                           how='left', on=['District'])

        # Drop object Column District   
        temp.drop(['District'], axis=1, inplace=True)

        min_y= -23.8
        max_y= -23.2
        min_x= -46.95
        max_x= -46        
             
        # Remove outliers and replace as NAN
        temp['Latitude'][(temp['Latitude'] < min_y )  |
                   (temp['Longitude'] < min_x )  |
                   (temp['Latitude']  > max_y )  |
                   (temp['Longitude'] > max_x )  
                   ]= np.nan

        temp['Longitude'][(temp['Latitude'] < min_y ) |
                           (temp['Longitude'] < min_x )  |
                           (temp['Latitude']  > max_y )  |
                           (temp['Longitude'] > max_x )  
                           ]= np.nan
        
        # Input into the NA values the mean point of the considered neigborhood
        temp.Latitude = np.where(temp.Latitude.isnull()
                                  , temp.Latitude_district # If Latitude is null replace with Latitude_district
                                  , temp.Latitude # else, keep the original value
                                 )
        temp.Longitude = np.where(temp.Longitude.isnull()
                                          , temp.Longitude_district # If Longitude is null replace with Longitude_district
                                          , temp.Longitude # else, keep the original value
                                         )

        # Drop temporary Columns
        temp.drop(['Latitude_district', 'Longitude_district'], axis=1, inplace=True)
        
        return temp


# In[5]:


pipeline = Pipeline([
        
    #Data preprocessing
    ('Criando as colunas', pre_processing_transform()),
    
    # Scaling our data with StandardScaler
    ('escalonando', StandardScaler())
])


# In[40]:


# Fit and Apply the Pipeline
housing_train_transformed = pipeline.fit_transform(X_train)


# In[41]:


# Saving Pipeline
joblib.dump(pipeline, './preprocessed_pipeline.pkl')


# ### 3.2 Metrics
# 
# The metrics we are using to evaluate our regression problems are R² and RMSE:
# 
# ##### **Coefficient of Determination - R²**
# 
# $$R^2(y, \hat{y}) = 1 - \frac {\sum_{i=0}^{n-1}(y^{(i)}-\hat{y}^{(i)})^2}{\sum_{i=0}^{m-1}(y^{(i)}-\bar{y})^2}$$
# 
# ##### **Root Mean Squared Error (RMSE)**
# 
# $$RMSE = \sqrt{MSE} = \sqrt{\frac{\sum_{i=0}^{m-1}(y^{(i)}-\hat{y}^{(i)})^2}{m}}$$
# 
# 
# In order to complete our pipeline, we'll build a function to:
# 
#     - Load our preprocessed pipeline;
#     - Set the Grid Search to our cross validation;
#     - Fit the data;
#     - Compute the metrics (R² and RMSE) to evaluate the models as well as the residuals;
#     - Plot some graphics to examinate the models predictions and residuals;
#     - Returns the best model as well as the predictions and residuals both for train and test samples.

# In[6]:


# Function to evaluate the regression models we'll test
# Metrics to evaluate the models:
# R²: for its easy interpretability
# RSME: Because its based on root square mean errors and its measure is in the same dimension as the target (Price) 

def resultados_regressao(modelo, string_nome_modelo, parametros):

    # Load a pre processed pipeline
    loaded_preprocessed_pipeline = joblib.load('./preprocessed_pipeline.pkl')
    
    # Create a full pipeline
    full_pipeline = Pipeline([
            ('preprocessing', loaded_preprocessed_pipeline),
            (string_nome_modelo, modelo)
    ])


    # GridSearch with 5 folds
    grid = GridSearchCV(full_pipeline,parametros,cv=5, scoring = 'neg_mean_squared_error', return_train_score=True, verbose=1)

    # Trainning the model
    grid.fit(X_train, y_train)

    # Best model in the grid search
    best_model = grid.best_estimator_

    print(('Melhores parametros encontrados: \n',grid.best_params_))  
    
    # Print the mediam RSME of all folds for the model with the best hiperparameters
    n_folds = 5
    split_keys = [f'split{i}_test_score' for i in range(n_folds)]
    best_index = grid.best_index_

    rmse_scores = []

    for key in split_keys:
        neg_mse_score = grid.cv_results_[key][best_index]
        rmse_scores.append(np.sqrt(-neg_mse_score))

    best_rmse = np.mean(rmse_scores)
    best_rmse_std = np.std(rmse_scores)


    print(f'Best RMSE score of all folds IN TRAIN: {best_rmse} +- {best_rmse_std}')
    
    
    # 1 - Getting the predictions
    y_train_pred = best_model.predict(X_train)
    
    y_teste_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_teste_pred)
    
    print(f'Best R² score of all folds IN TRAIN: {r2}')
    
    
    # 2 - Getting the residuals
    residual = y_train_pred - y_train
    
    residual_test =  y_teste_pred - y_test
    

    fig, axes = plt.subplots(3, 2, figsize=(18,14))

    sns.scatterplot(x=y_train_pred, y=y_train, ax=axes[0, 0], color = 'tab:blue')
    axes[0, 0].set_title('Housing value - Prediction vs Real - Regression (train)')

    sns.scatterplot(x=y_train_pred, y=residual, ax=axes[0, 1], color = 'tab:blue')
    axes[0, 1].set_title('Housing value - Real vs Residual - Regression (train)')


    sns.scatterplot(x=y_teste_pred, y=y_test, ax=axes[1, 0], color = 'tab:orange')
    axes[1, 0].set_title('Housing value - Prediction vs Real - Regression (test)')

    sns.scatterplot(x=y_teste_pred, y=residual_test,  ax=axes[1, 1], color = 'tab:orange')
    axes[1, 1].set_title('Housing value - Real vs Residual - Regression (test)')


    sns.boxplot(y=residual, ax=axes[2, 0], color= 'tab:blue')
    axes[2, 0].set_title('Boxplot - Residual (train)')

    sns.boxplot(y=residual_test, ax=axes[2, 1], color= 'tab:orange')
    axes[2, 1].set_title('Boxplot - Residual (test)')
    
    return best_model, y_train_pred, y_teste_pred, residual, residual_test;


# #### 3.3 Random Forest Regressor

# In[43]:


modelo_2 = RandomForestRegressor()

string_nome_modelo = 'random_forest'

parametros = [{string_nome_modelo+'__min_samples_leaf':[3,8],
              string_nome_modelo+'__min_samples_split':[10,50,80],
              string_nome_modelo+'__n_estimators':[100,250,500],
              string_nome_modelo+'__random_state': [42]
}]


# In[44]:


best_model_2, y_train_pred_2, \
    y_test_pred_2, residual_2, residual_test_2 = resultados_regressao(modelo_2, string_nome_modelo, parametros)


# In[86]:


r2 = r2_score(y_test, y_test_pred_2)
    
print(f'R² score IN TEST: {r2}')

rmse_test = mean_squared_error(y_test, y_test_pred_2, squared=False)

print(f'RMSE IN TEST: {rmse_test}')


# In[385]:


# Saving Best Random Forest regression model
joblib.dump(best_model_2, './best_random_forest.pkl')


# #### 3.4 Ada Boost Regressor

# In[375]:


modelo_3 = AdaBoostRegressor()

string_nome_modelo = 'AdaBoostRegressor'

parametros = [{
    
    string_nome_modelo+'__base_estimator': [DecisionTreeRegressor(max_depth = 10), RandomForestRegressor()] , 
    string_nome_modelo+'__random_state': [0],
    string_nome_modelo+'__n_estimators': [50,100] ,
    string_nome_modelo+'__learning_rate': [0.5,1,5] ,
    string_nome_modelo+'__loss': ['linear','square']
}]

best_model_3 , y_train_pred_3, \
    y_test_pred_3, residual_3, residual_test_3= resultados_regressao(modelo_3, string_nome_modelo, parametros)


# In[91]:


r2 = r2_score(y_test, y_test_pred_3)
    
print(f'R² score IN TEST: {r2}')

rmse_test = mean_squared_error(y_test, y_test_pred_3, squared=False)

print(f'RMSE IN TEST: {rmse_test}')


# In[386]:


# Saving Best Ada Boost regression model
joblib.dump(best_model_3, './best_adaboost.pkl')


# ## 🔍 4. Results
# 
# Finally, let's check on the results we've got to solve this regression problem.

# ### 4.1 Baseline
# 
# Before talking about any results, let's set our baseline.
# 
# Let's remind that currently, the **housing prices** at Seu Barriga are estimated ***manually by experts***: a team gathers up-to-date information about an apartment and finds out the _housing price_. 
# This is _costly_ and _time-consuming_, and their **estimates are not that great**; they often realize that **their estimates were off by more than 20%**.
# 
# In order to set this baseline, let's get random numbers from 20% to 25%.

# In[8]:


np.random.seed(42)

y_train_pred_baseline = []

for housing_price in y_train:
    percent_error = 1 + np.random.randint(20, 26) / 100
    y_train_pred_baseline.append(housing_price * percent_error)    
    
r2 = r2_score(y_train, y_train_pred_baseline)
print(f'R² score Baseline IN TRAIN: {r2}')

baseline_rmse_train = mean_squared_error(y_train, y_train_pred_baseline, squared=False)
print(f'RMSE Baseline IN TRAIN: {baseline_rmse_train}')    

y_test_pred_baseline = []

for housing_price in y_test:
    percent_error = 1 + np.random.randint(20, 26) / 100
    y_test_pred_baseline.append(housing_price * percent_error)

r2 = r2_score(y_test, y_test_pred_baseline)
print(f'R² score Baseline IN TEST: {r2}')

percent_error_analysts = ((y_test_pred_baseline - y_test)/y_test)*100

print(f'O erro percentual médio dos analistas em treino foi: {percent_error_analysts.abs().mean()}')


# ### 4.2 Discussion and results

# ### 4.2.1 Comparison of model metrics (and baseline)
# 
# As we've said before, we've trained many different models to solve this problem. 
# 
# Above we've a table with the mean of the **RMSE** in the 5 folds we've set for cross validation, considering the best parameters found in the Grid Search, for each model, in descending order.
# 
# 
# | MODEL | RMSE |
# | ----- | ---- | 
# | Ada Boost Regressor | 210.084 |
# | Gradient Boost Regressor | 229.465 |
# | Random Forest Regressor | 232.099 |
# | K Neighbors Regressor | 322.068 | 
# | Ridge Linear Model | 382.405 |
# | Bayesian Ridge | 382.428 |  
# | Lasso Linear Model | 388.374 |
# | Support Vector Regressor | 640.521 |
# 
# 
# From now one, we have chosen the **Ada Boost Regressor** and the **Random Forest Regressor** to look foward into the residual analysis.
# 
# 
# Above we've two tables with the metrics in Train and Test set, considering the baseline:
# 
# TRAIN 
# 
# | MODEL | RMSE | R² | 
# | ----- | ---------- | --------- |
# | Baseline - Analysts Price | 218.748 | 0.916 |
# | Ada Boost Regressor | 210.084 | 0.997 |
# | Random Forest Regressor | 232.010 | 0.960 |
# 
# 
# 
# TEST
# 
# | MODEL | RMSE | R² | 
# | ----- | ---------- | --------- |
# | Baseline - Analysts Price | 208.432 | 0.912 |
# | Ada Boost Regressor | 196.459 | 0.922 |
# | Random Forest Regressor | 230.837 | 0.892 |
# 
# 
# As we can see, our champion model should be **Ada Boost Regressor**, given it is the model we've achieved the best metrics performances. 
# 
# **Also, we've achieved better results than the standard baseline the company has actually**.
# 
# For Ada Boost Regressor, we've done the Grid Search on the space described bellow:
#    - base_estimator: [DecisionTreeRegressor(max_depth = 10), RandomForestRegressor()]
#    - n_estimators: [50,100]
#    - learning_rate: [0.5,1,5]
#    - loss: ['linear','square']
#    
# In this case, the best combination of parameters were:
#    - base_estimator: RandomForestRegressor()
#    - n_estimators: 50
#    - learning_rate: 0.5
#    - loss: square

# ### 4.2.2 Benefits and Results to Business
# 
# As we've just seen, our Ada Boost Regression model have had a performance in the **RMSE** and **R²** metrics that have beaten even the analysts baseline, which is already something to be really proud about! 
# 
# That's not the only benefit of our machine learning model to Seu Barriga Housing, and we'll list below all the implications by launching this model in production:
# 
# 
# | Challenge | Current Scenario | New Scenario's Proposal |
# | ----------------- | ---------------- | ----------------------- | 
# | Housing Price's deffinition | manually set by analists | Ada Boost Price's Prediction |
# | mean of percentual error (absolute) when testing the model's performance | 22.60% | 14.53% |
# | 10 Experts Analysts's role | Both Setting Prices on Houses and attending clients | Foccused on the clients needs and attending |
# | Response Time | At least 1 or 2 days to respond each client | 100% of clients response of Housing Prices at the same day (real time) |
# | Follow Up in Negotiations | poorly attended by the analysts, consumed in labor due to the onerous task of setting House Prices | As the analysts are now foccused on clients attending, they'll be abble to keep their follow up on the negotiations |
# | Scalability | Limitation of 500 clients attended a month | Within the new business model, they expect to be able to attend at least 750 clients per month (50% higher than before) |
# | NPS | 60 | Their new goal is to achieve NPS of 80 within the next 6 months |

# ### 4.3 Residual and Percent Error Analysis

# #### 4.3.1 Random Forest

# In[376]:


# Create a dataframe to evaluate the predict

results_test_2 = pd.DataFrame({'residual_test':residual_test_2, 'y_test':y_test,'y_teste_pred':y_test_pred_2   })

results_test_2['percent_error'] = (results_test_2['residual_test'] /results_test_2['y_test']) * 100

results_test_2 = pd.concat([X_test[['District', 'Latitude', 'Longitude']], results_test_2], axis=1)

fig = plt.subplots(figsize=(12,6))

sns.histplot(results_test_2.residual_test)
plt.title('Histogram - Percent Error {}');


# In[377]:


fig, axes = plt.subplots(1, 2, figsize=(12,6))

sns.histplot(results_test_2.percent_error, bins=200, ax=axes[0]);
axes[0].set_title('Histogram - Percent Error')

sns.scatterplot(x=results_test_2.percent_error, y=results_test_2.y_test, ax=axes[1])
axes[1].set_title('Scatterplot - Percent Error')


# In[383]:


# Let's take a look at the house where the percent error is greater then 800
results_test_2[results_test_2['percent_error']>800]


# In[384]:


X_test[431:432]


# #### 4.3.2 Ada Boost

# In[379]:


# Create a dataframe to evaluate the predict

results_test_3 = pd.DataFrame({'residual_test':residual_test_3, 'y_test':y_test,'y_teste_pred':y_test_pred_3 })

results_test_3['percent_error'] = (results_test_3['residual_test'] /results_test_3['y_test']) * 100

results_test_3 = pd.concat([X_test[['District', 'Latitude', 'Longitude']], results_test_3], axis=1)

fig = plt.subplots(figsize=(12,6))

sns.histplot(results_test_3.residual_test)
plt.title('Histogram - Percent Error');


# In[380]:


fig, axes = plt.subplots(1, 2, figsize=(12,6))

sns.histplot(results_test_3.percent_error, bins=200, ax=axes[0]);
axes[0].set_title('Histogram - Percent Error')

sns.scatterplot(x=results_test_3.percent_error, y=results_test_3.y_test, ax=axes[1])
axes[1].set_title('Scatterplot - Percent Error')


# ### 4.4 Residual per District
# 
# As we can see, both the Random Forest and the Ada Boost have had only 6 districts with mean of percent error greater then 20%. 
# 
# The 6 districts where this happened were the same in both models: Aricanduva, Rio Pequeno, Saúde, Perdizes, Ipiranga and Butantã.

# #### 4.4.1 Random Forest

# In[381]:


# Random Forest - Listing the 15 districts where the model performed worsely

temp = results_test_2.groupby(['District']).apply(lambda x: pd.Series(dict(
                                                        qtd_housing          = ((x.District.count()))
                                                        ,mean_percent_error_test   =  ((x.percent_error.mean())) 
                                                        ,median_price   =  ((x.y_test.median())) 
                                                        ))).reset_index() 

temp.sort_values(['mean_percent_error_test'], ascending = [False])[:15].style.background_gradient(cmap='Oranges', axis =0)


# #### 4.4.2 Ada Boost

# In[382]:


# Ada Boost - Listing the 15 districts where the model performed worsely

temp = results_test_3.groupby(['District']).apply(lambda x: pd.Series(dict(
                                                        qtd_housing          = ((x.District.count()))
                                                        ,mean_percent_error_test   =  ((x.percent_error.mean())) 
                                                        ,median_price   =  ((x.y_test.median())) 
                                                        ))).reset_index() 

temp.sort_values(['mean_percent_error_test'], ascending = [False])[:15].style.background_gradient(cmap='Oranges', axis =0)


# In[7]:


#! pip install streamlit

#! pip install geopy 

#! pip install folium


# ## 4.5 Deploy
# 
# 
# All these notebook was made as a small abstract of our work, you can see more in :
# 
#    - [Sprint 1](https://github.com/carlostomeh/Predicao_Preco_Apto_Sao_Paulo/blob/main/sao-paulo-housing__sprint1.ipynb)
#    - [Sprint 2](https://github.com/carlostomeh/Predicao_Preco_Apto_Sao_Paulo/blob/main/sao-paulo-housing__sprint2.ipynb)
#    - [Sprint 3](https://github.com/carlostomeh/Predicao_Preco_Apto_Sao_Paulo/blob/main/sao-paulo-housing__sprint3.ipynb)
#     
#     
# You can also use this local app running our machine learning solution:
# 
#    - [Machine Learning App](https://github.com/carlostomeh/Predicao_Preco_Apto_Sao_Paulo/blob/main/my_app.py)
# 
# Or running code below:
#     
#     

# In[5]:


get_ipython().system(' streamlit run my_app.py')


# ## 5 Appendix 
# 
# In our app we use a function to find the most similar apartments to an specified by the user, for this we use an method to find the distance between the features to each sample in our dataset. We use a customized euclidean distance that sets diferent weights to each feature. Below we list the features and each weight attributed: 
# 
# 
# Features use:
# 
# - Size; Weight=1
# - Condo; Weight=1
# - Total Rooms; Weight=1
# - Total Bedrooms; Weight=1
# - Price; Weight=2
# 
# 
# ### Formula
# 
# ##### **Euclidean distance with weights**
# 
# $$D(y, \hat{y}) =  \sqrt {{\sum_{i=0}^{n-1}((y^{(i)}-\hat{y}^{(i)})^2) * {w}^{(i)}}}$$

# In[8]:


def busca_apartamentos_similares(X_all, df): 
    
    '''
    Função que retorna dataframe principal para os 10 apartamentos mais próximos segundo o racional da distancia euclideana com pesos
    Recebe como input o dataframe X_all com todos os dados semi limpos e transforma em:
    
    input_transformed = Dataframe normalizado da requisição do usuario
    df_transformed    = Dataframe X_all limpo e normalizado
    
    Retorna um dataframe com os 10 vizinhos mais proximos por distancia euclideana.
    
    '''
    
    pipeline_novo = Pipeline([  ('Criando as colunas', pre_processing_transform_cluster()),
                            ('escalonando', StandardScaler()) ])

    # transform all data
    df_transformed = pipeline_novo.fit_transform(X_all)

    # user data
    input_transformed = pipeline_novo.transform(df)

                

    # Cria lista vazia
    list_distance = []

    # Itera para as colunas de interesse
    for i in range(0, len(df_transformed)):

        # Calcula as distancias para cada feature
        distancia_1 = (df_transformed[i][0] - input_transformed[0][0]) **2
        distancia_2 = (df_transformed[i][1] - input_transformed[0][1]) **2
        distancia_3 = (df_transformed[i][2] - input_transformed[0][2]) **2
        distancia_4 = (df_transformed[i][3] - input_transformed[0][3]) **2
        distancia_5 = (df_transformed[i][4] - input_transformed[0][4]) **2

        # Cria a medidade de distancia - ( Com pesos )
        distance_temp = (distancia_1 + distancia_2 + distancia_3 + distancia_4 + (2*distancia_5))**(0.5)        

        list_distance.append(distance_temp)


    # Cria coluna de distancia ao centroide

    X_all['distancia_centroide'] = list_distance
    mais_proximos = X_all.dropna(subset=['Latitude', 'Longitude']).sort_values(['distancia_centroide'], ascending = [True])[0:10]

    mais_proximos['Total_Rooms']            = mais_proximos['Rooms'] + mais_proximos['Toilets'] + mais_proximos['Suites']
    mais_proximos['Total_Bedrooms']         = mais_proximos['Rooms'] + mais_proximos['Suites']
    
    return mais_proximos;

