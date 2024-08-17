#!/usr/bin/env python
# coding: utf-8

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print((os.path.join(dirname, filename)))

# Any results you write to the current directory are saved as output.


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import normaltest
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')


# In[12]:


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# In[13]:


#create class
class car_price_model:
    """  **car_price_model** is the class for exploratory data 
        analysis and machine learning in each data car. 
        This class have 10 attributes that are given important:

    - multi_categorical_plot

    - distplot_multi

    - boxplot_multi

    - correlation_plot

    - VIF

    - learner_selection

    - training_evaluate
    """
    
    def __init__(self, data=None, cols=None, name='price'):
        
        self.name = name # target
        self.data = data # feature
        self.cols = cols # feature columns name
        self.listof_model = {'LinearRegression': LinearRegression(), 
                'KNeighborsRegression':KNeighborsRegressor(),
                'RandomForestRegression': RandomForestRegressor(),
               'GradientBoostingRegression': GradientBoostingRegressor(),
                'XGBoostRegression': XGBRegressor(),
                'adaboost':AdaBoostRegressor()} # list of different learner
    
    #Read csv file
    def read(self, file):
        return pd.read_csv(file)
    
    def multi_categorical_plot(self, data):
    
        """ plot a categorical feature
        
            data: float64 array  n_observationxn_feature
        
        """
        # Find a feature that type is object
        string = []
        for i in data.columns:
            if data[i].dtypes == "object":
                string.append(i)
    
        fig = plt.figure(figsize=(20,5))
        fig.subplots_adjust(wspace=0.2, hspace = 0.3)
        for i in range(1,len(string)+1):
            ax = fig.add_subplot(2,3,i)
            sns.countplot(x=string[i-1], data=data, ax=ax)
            ax.set_title(f" {string[i-1]} countplot")
            
    def distplot_multi(self, data):
        """ plot multi distplot"""
    
        
        from scipy.stats import norm
        cols = []
        
        #Feature that is int64 or float64 type 
        for i in data.columns:
            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':
                cols.append(i)
        
        gp = plt.figure(figsize=(15,10))
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(2,3,i)
            sns.distplot(data[cols[i-1]], fit=norm, kde=False)
            ax.set_title('{} max. likelihood gaussian'.format(cols[i-1]))
            
    def boxplot_multi(self, data):
        
        """ plot multi box plot
            hue for plotting categorical data
        """
    
        cols = []
        for i in data.columns:
            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':
                cols.append(i)
    
        gp = plt.figure(figsize=(15,10))
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(2,3,i)
            sns.boxplot(x = cols[i-1], data=data)
            ax.set_title('Boxplot for {}'.format(cols[i-1]))
            
    def correlation_plot(self, data, vrs= 'price'):
    
        """
        This function plot only a variable that are correlated with a target  
        
            data: array m_observation x n_feature
            vrs:  target feature (n_observation, )
            cols: interested features
        """
        
        cols = []
        for i in data.columns:
            if data[i].dtypes == "float64" or data[i].dtypes == 'int64':
                cols.append(i)
                
        feat = list(set(cols) - set([vrs]))
    
        fig = plt.figure(figsize=(15,10))
        fig.subplots_adjust(wspace = 0.3, hspace = 0.25)
        for i in range(1,len(feat)+1):
        
            gp = data.groupby(feat[i-1]).agg('mean').reset_index()
        
            if len(feat) < 3:
                ax = fig.add_subplot(1,3,i)
            else:
                n = len(feat)//2 + 1
                ax = fig.add_subplot(2,n,i)
            
            ax.scatter(data[feat[i-1]], data[vrs], alpha=.25)
            ax.plot(gp[feat[i-1]], gp[vrs], 'r-', label='mean',  linewidth=1.5)
            ax.set_xlabel(feat[i-1])
            ax.set_ylabel(vrs)
            ax.set_title('Plotting data {0} vs {1}'.format(vrs, feat[i-1]))
            ax.legend(loc='best')
            
    # Standardize data
    def standardize(self, data):
        data = (data - data.mean())/data.std()
        return data
            
            
    def VIF(self, data):
        """ 
        This function compute variance inflation factor for data that all feature are multicolinear
        
        if the outcome is 1, it is okay
        if it is between 1 and 5, it shows low to average colinearity, and above 5 generally means highly 
        redundant and variable should be dropped
        """ 
        # Apply the standardize method to each feature and save it to a new data
        std_data = data.apply(self.standardize, axis=0)
    
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    
        vif = pd.DataFrame()
        vif['VIF_FACTOR'] = [variance_inflation_factor(std_data.values, i) for i in range(std_data.shape[1])]
    
        vif['feature'] = std_data.columns
    
        return vif
    
    
    def split_data(self):
        """
        This function splits data to train set and target set
        
        data: matrix feature n_observation x n_feature dimension
        name: target  (n_observation, )
        cols: interested feature
        
        return xtrain, xtest, ytrain, ytest
        """
    
        train = self.data[self.cols]
        target = self.data[self.name]
    
        return train_test_split(train, target, random_state=42, test_size=0.2, shuffle=True)
    
    def spearman_pearson_correlation(self, data):
        
        
        gp = plt.figure(figsize=(15,5))
        cols = ['pearson', 'spearman']
        gp.subplots_adjust(wspace=0.4, hspace=0.4)
        for i in range(1, len(cols)+1):
            ax = gp.add_subplot(1,2,i)
            sns.heatmap(data.corr(method=cols[i-1]), annot=True)
            ax.set_title('{} correlation'.format(cols[i-1]))
        
        
        plt.show()
    
    
    def learner_selection(self):

        """
            This function compute differents score measure like cross validation,
            r2, root mean squared error and mean absolute error.
            listof_model: dictionary type containing different model algorithm.     
        """ 
    
        result = {}
        
        x, _, y, _ = self.split_data() # take only xtrain and ytrain
    
        for cm in list(self.listof_model.items()):
        
            name = cm[0]
            model = cm[1]
        
            cvs = cross_val_score(model, x, y, cv=10).mean()
            ypred = cross_val_predict(model, x, y, cv=10)
            r2 = r2_score(y, ypred)
            mse = mean_squared_error(y, ypred)
            mae = mean_absolute_error(y, ypred)
            rmse = np.sqrt(mse)
        
            result[name] = {'cross_val_score': cvs, 'rmse': rmse, 'mae': mae, 'r2': r2}
        
            print(('{} model done !!!'.format(name)))
        
        
        return pd.DataFrame(result)
    
    
    def training_evaluate(self, algorithm):
        
        """This function train and evaluate our model to find r2, rmse and mae"""
        
        result = {}
        xtrain, xtest, ytrain, ytest = self.split_data()
        
        learner = self.listof_model[algorithm] # learner selected in model_selection function
        
        model = learner.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        
        r2 = learner.score(xtest, ytest)
        rmse =  np.sqrt(mean_squared_error(ytest, ypred))
        mae = mean_absolute_error(ytest, ypred)
        
        result['car price measure'] = {'r2':round(r2, 3),  'rmse':round(rmse, 3), 'mae':round(mae, 3)}
        
        return  pd.DataFrame(result)
        


# ### car_price_model class explaination
# 
# **car_price_model** is the class that I use to do exploratory data analysis and machine learning in each data car. This class have 10 attributes that are:
# 
# - multi_categorical_plot
# 
# - distplot_multi
# 
# - boxplot_multi
# 
# - spearman_pearson_correlation
# 
# - correlation_plot
# 
# - VIF
# 
# - learner_selection
# 
# - training_evaluate
# 
# These are the function that we are going to use in this notebook. 
# 
# **N.B: We are making prediction price for 5 cars (focus, audi, ford, toyota, skoda)** 

# In[14]:


car1 = '/kaggle/input/used-car-dataset-ford-and-mercedes/focus.csv'
car2 = '/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv'
car3 = '/kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv'
car4 = '/kaggle/input/used-car-dataset-ford-and-mercedes/toyota.csv'
car5 = '/kaggle/input/used-car-dataset-ford-and-mercedes/skoda.csv'


# In[15]:


model = car_price_model()


# # Focus car price

# In[16]:


focus = model.read(car1)


# In[17]:


focus.head()


# In[18]:


focus.info()


# ## Visualization, correlation, VIF, learner selection, training and evaluation

# In[19]:


model.multi_categorical_plot(focus)


# In[20]:


sns.countplot(x = 'model', hue='fuelType', data=focus)


# In[21]:


focus.describe()


# In[22]:


model.distplot_multi(focus)


# In[23]:


model.boxplot_multi(focus) # we see well that our maximun likelihood gaussian go with our boxplot. 


# In[24]:


model.spearman_pearson_correlation(focus)


# In[25]:


model.correlation_plot(focus)


# with this two correlations we can see that **price is most correlated with mileage and year**. also year well correlated with mileage. We use VIF to see how this correlation are.

# In[26]:


focus_cols = ['mileage', 'year', 'engineSize'] #take columns


# In[27]:


model.VIF(focus[focus_cols])


# In[28]:


focus_model = car_price_model(data=focus, cols=focus_cols) #select best algorithm


# In[29]:


focus_model.learner_selection()


# In[30]:


focus_model.training_evaluate('GradientBoostingRegression')


# **We have** $R^2 = 92.1\%$ **for Focus car**

# # Audi car price

# In[31]:


audi = model.read(car2)


# In[32]:


audi.head()


# In[33]:


audi.info()


# ## Visualization, correlation, VIF, learner selection, training and evaluation

# In[34]:


model.multi_categorical_plot(audi)


# In[35]:


audi.describe()


# In[36]:


model.distplot_multi(audi)


# In[37]:


model.boxplot_multi(audi)


# In[38]:


model.spearman_pearson_correlation(audi)


# In[39]:


model.correlation_plot(audi)


# In[40]:


audi_cols = ['year', 'mileage', 'mpg', 'engineSize', 'tax'] #there is or not necessary to take tax feature


# In[41]:


model.VIF(audi[audi_cols]) #all vif factor of feature are acceptable


# In[42]:


#select learner
audi_model = car_price_model(data=audi, cols=audi_cols)


# In[43]:


audi_model.learner_selection()


# In[44]:


audi_model.training_evaluate('XGBoostRegression')


# **We have** $R^2 = 94.5\%$ **for Audi car**

# # Ford car price

# In[45]:


ford = model.read(car3)


# In[46]:


ford.head()


# In[47]:


ford.info()


# In[48]:


ford = ford.replace(to_replace=2060, value=2016) #some errors


# ## Visualization, correlation, VIF, learner selection, training and evaluate

# In[49]:


model.multi_categorical_plot(ford) 


# In[50]:


ford.describe()


# In[51]:


model.distplot_multi(ford)


# In[52]:


model.boxplot_multi(ford)


# In[53]:


model.spearman_pearson_correlation(ford)


# In[54]:


model.correlation_plot(ford)


# In[55]:


ford_cols = ['mileage', 'year', 'tax', 'engineSize', 'mpg']


# In[56]:


model.VIF(ford[ford_cols])


# In[57]:


ford_model = car_price_model(data=ford, cols=ford_cols)


# In[58]:


ford_model.learner_selection()


# In[59]:


ford_model.training_evaluate('XGBoostRegression')


# **We get** $R^2 = 91.6\%$ **for Ford car**

# # Toyota car price

# In[60]:


toyota = model.read(car4)


# In[61]:


toyota.head()


# In[62]:


toyota.info()


# ## Visualization, correlation, VIF, learner selection, training and evaluate

# In[63]:


model.multi_categorical_plot(toyota)


# In[64]:


toyota.describe()


# In[65]:


model.distplot_multi(toyota)


# In[66]:


model.boxplot_multi(toyota)


# In[67]:


model.spearman_pearson_correlation(toyota)


# In[68]:


model.correlation_plot(toyota)


# In[69]:


toyota_cols = ['engineSize','year','tax', 'mileage', 'mpg']


# In[70]:


model.VIF(toyota[toyota_cols])


# In[71]:


toyota_model = car_price_model(data=toyota, cols=toyota_cols)


# In[72]:


toyota_model.learner_selection()


# In[73]:


toyota_model.training_evaluate('XGBoostRegression')


# **We obtain** $R^2 = 96.2\%$ **for Toyota car.**

# # Skoda car price

# In[74]:


skoda = model.read(car5)


# In[75]:


skoda.head()


# In[76]:


skoda.info()


# ## Visualization, correlation, VIF, learner selection, training and evaluate

# In[77]:


model.multi_categorical_plot(skoda)


# In[78]:


skoda.describe()


# In[79]:


model.distplot_multi(skoda)


# In[80]:


model.boxplot_multi(skoda)


# In[81]:


model.spearman_pearson_correlation(skoda)


# In[82]:


model.correlation_plot(skoda)


# In[83]:


skoda_cols = ['year', 'engineSize', 'mileage', 'tax', 'mpg']


# In[84]:


model.VIF(skoda[skoda_cols])


# In[85]:


skoda_model = car_price_model(data=skoda, cols=skoda_cols)


# In[86]:


skoda_model.learner_selection()


# In[87]:


skoda_model.training_evaluate('XGBoostRegression')


# **We obtain** $R^2 = 92.6\%$ **for Skoda car.**

# ### Summarize
# 
# For this five cars, we obtain de $R^2$ score:
# 
# > Skoda car $92.6\%$
# 
# > Toyota car $96.2\%$
# 
# > Ford car $91.6\%$
# 
# > Audi car $94.5\%$
# 
# > Focus car $92.1\%$
# 
# **For the rest of car, you can continue to find a score. If you do not understand my model or method be free to ask your question.**

# # Update

# In[ ]:




