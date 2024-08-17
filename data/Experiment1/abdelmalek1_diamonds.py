#!/usr/bin/env python
# coding: utf-8

# ## What is the Daimond ?
# It's a precious stone consisting of a clear and colourless crystalline form of pure carbon, the hardest naturally occurring substance.
# 
# ## Why are the diamonds so rare ? 
# Diamonds are one of the hardest materials found on Earth. Other than that, they hold no unique distinctions. All gem grade materials are rare, composing just a tiny fraction of the Earth. However, among gems, diamonds are actually the most common.
# 
# ## Objective 
# We need to predict the value of the diamonds from their features
# 
# ## Why the problem need to be solved 
# The price of the diamond is hard to figure for normal buyer as it depends on many factors you can see two diamonds with ths same weight(carat) but stil very diffferent on price scale .  
# " Take a look at [this diamond from Amazon](https://www.amazon.com/gp/product/B072L343M2/ref=as_li_qf_asin_il_tl?ie=UTF8&tag=thediamondpro-20&creative=9325&linkCode=as2&creativeASIN=B072L343M2&linkId=b169c65833f9b1074b4c82cb125f7e58) and then take a look this [diamond from Blue Nile](https://www.bluenile.com/diamond-details/LD09889381?click_id=962062758). They’re both one carat diamonds. Does it make sense that they’re the same size, yet one costs \$1,179 and the other \$16,500? To make it even crazier, the Blue Nile diamond may actually be a better value!" [The diamonds pro](https://www.diamonds.pro/education/diamond-prices/)  
# 
# To avoid being scammed or failed to have a good baragain, a good price prediction may come in handy 

# ## Topics 
# 0. Loading Libraries and helping functions 
# 1. Exploring Dataset 
# 2. Analysis 
# 3. Preprocessing 
# 4. Feature Engineering 
# 5. Model 
# 6. Tuning 
# 7. Evaluation 

# ## 0) Loading libraries and helping functions 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O and data manipulation 

#visulaizations 
import matplotlib.pyplot as plt   
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# let's define some helping functions 
def cor_map(df):
    ## objective : drawing a heating map of correlation between the numerical features and each other
    ## input : data 
    
    cor=df.corr()
    _,ax=plt.subplots(figsize=(12,10))
    cmap=sns.diverging_palette(192,6,as_cmap=True)
    _=sns.heatmap(cor,cmap=cmap,ax=ax,square=True,annot=True)
    
def print_coef(model,x):
    ## objective : printing the coefficients of each feature to determine it's importance 
    ## input: ML model , X(all data except the target variable) 
    coeff_df = pd.DataFrame(x.columns)
    coeff_df.columns = ['Variable']
    coeff_df["Coeff"] = pd.Series(model.coef_)
    coeff_df.sort_values(by='Coeff', ascending=True)
    print(coeff_df)
    
def metrics(y_true,y_pred):
    ## objective: printing the R^2, mean squared error and mean absolute error 
    ## input: the actual target variable Series and the predicted target variable Series 
    print(("R2 score:",r2_score(y_true,y_pred)))
    print(('mean squared error',mean_squared_error(y_true,y_pred)))
    print(("mean absolute error",mean_absolute_error(y_true,y_pred)))


# In[ ]:


# loading the data 
data=pd.read_csv('../input/diamonds.csv')
data.head()


# In[ ]:


# removing the unwanted column 
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.head()


# In[ ]:


# show the classes of every categorical feature 
for i in data.select_dtypes(include=['O']).columns:
    print((i,data[i].unique()))


# ## Features:
# * **carat**: Diamond carat weight is the measurement of how much a diamond weighs. A metric “carat” is defined as 200 milligrams
# * **cut**: diamond cut is a style or design guide used when shaping a diamond for polishing,  
#     it's classes from best to worst are : Ideal,Premium,Very Good, Good,Fair
# 
# * **color**: the color of the diamond, The less body color in a white diamond, the more true color it will reflect, and thus the greater its value  
#     it's classes from best to worst: D,E,F,G,H,I,J
# 
# * **clarity**:Clarity refers to the degree to which these imperfections are present. Diamonds which contain numerous or significant inclusions or blemishes have less brilliance because the flaws interfere with the path of light through the diamond.  
#     it's classes from the best to the worst: IF, VVS2,VVS1,VS2,VS1,SI2,SI1,I1
# 
# * **depth**: depth % :The height of a diamond, measured from the culet to the table, divided by its average girdle diameter
# 
# * **x**,**y**,**z**: Length,width,height
# * **price**: Is the price of the diamond
# * **table**: table%: The width of the diamond's table expressed as a percentage of its average diameter
# <img src="https://www.lumeradiamonds.com/images/diamond-education/depth_table.gif" />

# In[ ]:


#check data types and  if there's no missing data
data.info()


# **There is no missing data, we have 7 numerical features and 3 categorical features **

# ## analysis

# In[ ]:


x=data.drop('price',axis=1)
y=data['price']


# In[ ]:


data.describe()


# In[ ]:


data.describe(include=['O'])


# * **depth and table have mean nearly equal to the median, which indicates that their distribution is nearly normal with small standard deviation --> this could means that these features independently could be with little effect on the target value since the variation is very small **
# * **carat and the three dimensions have high standard deviation which indicates that there is high variance in these features thus higher effection on the target value **
# * **the three dimensions has zero values !!!??? which is very weird it's impossible to have any dimension in diamond equal to zero so it must be error in entry (<span style='color:red'>complete feature</span>)**

# In[ ]:


cor_map(data)


# In[ ]:


print((data[['cut','price']].groupby('cut',as_index=False).mean().sort_values(by='price',ascending=False)))
print((data[['color','price']].groupby('color',as_index=False).mean().sort_values(by='price',ascending=False)))
print((data[['clarity','price']].groupby('clarity',as_index=False).mean().sort_values(by='price',ascending=False)))


# * **price has extremly high correlation with carat and dimensions, and that's somehow rational as the size of the diamond increases, it's price increases **
# * **as we predicted table and depth has small effect on the price, also they're negatively correlated, in other words somehow inversely proportional **
# * **all the relations between the categorical features and the price seem not reasonable, diamonds with fair cut has higher mean price than that with ideal cut, the same way with the colors and clarity, we could explain this by saying that other factors has much higher effect on the price than them, and since we know that the size of the diamond had the highest correlation rate among the numerical data, then maybe we need to generate a price for every carat(200 mg) of diamond.(<span style='color:indigo'>generate new feature</span>)**

# ## PreProcessing & Feature extraction 

# In[ ]:


#how many exactly missing data we have
data.loc[(data['x']==0)|(data['y']==0)|(data['z']==0)].shape[0]


# In[ ]:


# we will exclude them from the dataset since 20 aren't important
data=data.loc[(data['x']!=0)&(data['y']!=0)&(data['z']!=0.0)]


# In[ ]:


data['p/ct']=data['price']/data['carat']
print((data[['cut','p/ct']].groupby('cut',as_index=False).mean().sort_values(by='p/ct',ascending=False)))
print((data[['color','p/ct']].groupby('color',as_index=False).mean().sort_values(by='p/ct',ascending=False)))
print((data[['clarity','p/ct']].groupby('clarity',as_index=False).mean().sort_values(by='p/ct',ascending=False)))


# **so now officially we predict prict/carat and by multiplying it with carat it will return to price, or we could instead multipy each class by carat as a weight!!!**

# In[ ]:


data['cut']=data['cut'].map({'Ideal':1,'Good':2,'Very Good':3,'Fair':4,'Premium':5})
data['color']=data['color'].map({'E':1,'D':2,'F':3,'G':4,'H':5,'I':6,'J':7})
data['clarity']=data['clarity'].map({'VVS1':1,'IF':2,'VVS2':3,'VS1':4,'I1':5,'VS2':6,'SI1':7,'SI2':8})
data.head()


# In[ ]:


#also we can merge the thre dimensions into volume 
data['volume']=data['x']*data['y']*data['z']
data['table*y']=data['table']*data['y']
data['depth*y']=data['depth']*data['y']


# In[ ]:


data['cut/wt']=data['cut']/data['carat']
data['color/wt']=data['color']/data['carat']
data['clarity/wt']=data['clarity']/data['carat']


# In[ ]:


data.drop(['carat','cut','color','clarity','depth','table','x','y','z','p/ct'],axis=1,inplace=True)
cor_map(data)


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(data.drop(['price'],axis=1),data['price'],test_size=0.25,random_state=1)
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)
print(("The shape of the train set",X_train.shape))
print(("The shape of the test set",X_test.shape))


# ## Training and Evaluation 

# #### 1- Linear Regression 

# In[ ]:


reg_all=LinearRegression()
reg_all.fit(X_train_scaled,y_train) #fitting the model for the x and y train

pred=reg_all.predict(X_test_scaled) #predicting y(the target variable), on x test

# Rsquare=reg_all.score(X_test,y_test)
R2=r2_score(y_test,pred)
# print("Rsquare: %f" %(Rsquare))
# print("R2:",R2)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,reg_all.predict(X_train_scaled))
print_coef(reg_all,X_train)


#  #### 2- K nearest neighbor

# In[ ]:


kn_model=KNeighborsRegressor(n_neighbors=3)
kn_model.fit(X_train_scaled,y_train)
pred=kn_model.predict(X_test_scaled)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,kn_model.predict(X_train_scaled))


# #### 3- Gradient Boosting 

# In[ ]:


gbr = GradientBoostingRegressor(random_state=0)
gbr.fit(X_train_scaled,y_train)
pred=gbr.predict(X_test_scaled)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,gbr.predict(X_train_scaled))


# #### 4- XGBoost 

# In[ ]:


xgb = XGBRegressor(random_state=0,n_jobs=-1)
xgb.fit(X_train_scaled,y_train)
pred=xgb.predict(X_test_scaled)
print ('=========\nTest results')
metrics(y_test,pred)
print ('=========\nTrain results')
metrics(y_train,xgb.predict(X_train_scaled))


# ## Grid Search with cross validation

# In[ ]:


clf = XGBRegressor(random_state=0,n_jobs=-1)
cv_sets = ShuffleSplit(X_train.shape[0], n_iter =10,test_size = 0.20, random_state = 7)
parameters = {'n_estimators':list(range(100,1000,100)),
#              'max_depth':np.linspace(1,32,32,endpoint=True,dtype=np.int),
             'learning_rate':[0.05,0.1,0.25,0.5,0.75],}
#              'reg_lambda':[1,10,15,20,25]}
scorer=make_scorer(r2_score)
grid_obj=GridSearchCV(clf, parameters, scoring=scorer,verbose=1,cv=cv_sets)
grid_obj= grid_obj.fit(X_train_scaled,y_train)
clf_best = grid_obj.best_estimator_
print(clf_best)
clf_best.fit(X_train_scaled,y_train)


# In[ ]:


print(clf_best)
print ('=========\nTrain results')
metrics(y_train,clf_best.predict(X_train_scaled))
print ('=========\nTest results')
metrics(y_test,pred)


# In[ ]:





# In[ ]:




