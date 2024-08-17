#!/usr/bin/env python
# coding: utf-8

# In[78]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score


# In[79]:


df=pd.read_csv('train.csv')


# In[80]:


df.head()


# In[81]:


df.shape


# In[82]:


df.info()


# In[83]:


df.describe()


# ### considering 10% as a minimum missing value percentage threshold and removing everything above it

# In[84]:


list_to_drop=list(round(100*(df.isnull().sum()/len(df.index)),2)[round(100*(df.isnull().sum()/len(df.index)),2).values>10].keys())
list_to_drop


# In[85]:


df=df.drop(list_to_drop,axis=1)


# In[86]:


df.shape


# ### Checking the columns where the missing values between 0-10%

# In[87]:


round(100*(df.isnull().sum()/len(df.index)),2)[round(100*(df.isnull().sum()/len(df.index)),2).values>0]


# ### As we can see that Garagetype ,year build,finsh,qualtiy,condition have equal missing values ie 5.55 this indicates that there can be no garage in these houses so we should impute this value

# ### First of all we should deal with the Year columns and should replace with the old age

# In[88]:


df['YearBuilt_Age'] = df.YearBuilt.max()-df.YearBuilt
df['YearRemodAdd_Age'] = df.YearRemodAdd.max()-df.YearRemodAdd
df['GarageYrBlt_Age'] = df.GarageYrBlt.max()-df.GarageYrBlt
df['YrSold_Age'] = df.YrSold.max()-df.YrSold


# ### Dropping the actual year columns

# In[89]:


df = df.drop(['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],axis=1)


# ### Now imputing the missing values

# #### Similar to Garage there is a Basement columns having nearly equal missing percentage values ie 2.53 and by looking in the data dictionary there is "NA" where there is no basement so imputing the respective values in them
# 

# #### Imputing the values as per the data dictionary values for respective columns

# In[90]:


# NP refers to "NOT PRESENT"
df.MasVnrType.fillna('None',inplace=True)
df.MasVnrArea.fillna(df.MasVnrArea.mean(),inplace=True)
df.BsmtQual.fillna('NP',inplace=True)
df.BsmtCond.fillna('NP',inplace=True)
df.BsmtExposure.fillna('NP',inplace=True)
df.BsmtFinType1.fillna('NP',inplace=True)
df.BsmtFinType2.fillna('NP',inplace=True)
df.GarageType.fillna('NP',inplace=True)
df.GarageYrBlt_Age.fillna(0,inplace=True)
df.GarageFinish.fillna('NP',inplace=True)
df.GarageQual.fillna('NP',inplace=True)
df.GarageCond.fillna('NP',inplace=True)
df.Electrical.fillna(df.Electrical.mode()[0],inplace=True)


# In[91]:


df.skew()


# In[92]:


p_cat=df.columns[df.dtypes.values=='object']


# In[93]:


for i in p_cat:
    print((df[i].value_counts()))


# ### By looking the above data we can see that "Street" and "Utilities" are highly skewed so lets just remove these two columns
# 

# In[94]:


df=df.drop(['Street','Utilities'],axis=1)


# ### ID column is also of no use so dropping it

# In[95]:


df= df.drop('Id',axis=1)


# ## EDA

# ### plotting the graphs for numerical columns with SalePrice of Houses

# In[96]:


p=df.columns[df.dtypes.values!='object']


# In[97]:


p=p.drop('SalePrice')


# In[98]:


for i in p:
    figure=plt.figure()
    plt.scatter(df[i],df['SalePrice'])
    plt.xlabel(i)
    plt.ylabel('SalePrice')
    plt.title("{} vs SalePrice".format(i))
    plt.show()


# ### By looking at the above scatter plots , it is observed that many numerical columns are actually categorical columns containing discrete values,where as actual numerical columns are:
# ### LotArea,MasVnrArea,TotalBsmtSF,1stFlrSF,GarageArea,GrLivArea,WoodDeckSF,etc

# #### Lets also plot heatmap to check the correllations

# In[99]:


plt.figure(figsize = (30,20))        # Size of the figure
sns.heatmap(df.corr(),annot = True)
plt.show()


# ### By plotting the correllation matrix we see that "GrLiveArea" is highly correllated with "TotRmsAbvGrd" as well as TotalBsmtSf is also corr so lets drop both. Note: we are ignoring the correllations of SalePrice as it is our target variable

# In[100]:


df=df.drop('GrLivArea',axis=1)
df=df.drop('TotalBsmtSF',axis=1)


# ### Now lets plot the categorical variables with SalePrice

# In[101]:


p=df.columns[df.dtypes.values=='object']


# In[102]:


for i in p:
    figure=plt.figure()
    sns.boxplot(x = 'SalePrice', y =i, data = df)
    plt.ylabel(i)
    plt.xlabel('SalePrice')
    plt.title("{} vs SalePrice".format(i))
    plt.show()


# ### Lets check the distribution of our target varibale SalePrice

# In[103]:


plt.figure(figsize=(16,6))
sns.distplot(df.SalePrice)
plt.show()


# ### As we see that it is not normally distributed so lets apply log function on SalePrice

# In[104]:


plt.figure(figsize=(16,6))
sns.distplot(np.log(df.SalePrice))
plt.show()


# In[105]:


df['SalePrice']=df['SalePrice'].apply(lambda x:np.log(x))


# ### Creating Dummy Variables

# In[106]:


#we already have list of categorical columns ie p
dummy= pd.get_dummies(df[p],drop_first=True)
df= pd.concat([df,dummy],axis=1)
df = df.drop(p,axis='columns')


# In[107]:


y=df.pop('SalePrice')
X=df


# In[108]:


X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=42)


# In[109]:


#getting names of all columns except the dummy columns
for i in X_train.columns:
    print(i)


# In[110]:


num_col=['MSSubClass','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YearBuilt_Age','YearRemodAdd_Age','GarageYrBlt_Age','YrSold_Age']


# In[ ]:





# In[111]:


scaler=StandardScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])


# ### Doing Lasso regression first 

# In[112]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# cross validation
folds = 5

lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'r2', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train,y_train) 


# In[113]:


cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[114]:


plt.figure(figsize=(16,8))
plt.plot(cv_results['param_alpha'],cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'],cv_results['mean_test_score'])
plt.xscale('log')
plt.ylabel('R2 Score')
plt.xlabel('Alpha')
plt.show()


# In[115]:


model_cv.best_params_


# In[116]:


lasso = Lasso(alpha=0.001)
lasso.fit(X_train,y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print((r2_score(y_true=y_train,y_pred=y_train_pred)))
print((r2_score(y_true=y_test,y_pred=y_test_pred)))


# In[117]:


lasso.coef_


# In[118]:


# lasso model parameters
model_parameters = list(lasso.coef_)
model_parameters.insert(0, lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[119]:


model_param = list(lasso.coef_)
model_param.insert(0,lasso.intercept_)
cols = X_train.columns
cols=cols.insert(0,'const')
lasso_coef = pd.DataFrame(list(zip(cols,model_param)))
lasso_coef.columns = ['Feature','Coefficient']


# In[120]:


lasso_coef['Coefficient']=lasso_coef['Coefficient'].apply(lambda x :abs(x))


# In[121]:


lasso_coef.sort_values(by='Coefficient',ascending=False).head(10)


# ### Ridgre Regression

# In[122]:


folds  = KFold(n_splits=10,shuffle=True,random_state=42)

hyper_param = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
model = Ridge()

model_cv = GridSearchCV(estimator=model,
                        param_grid=hyper_param,
                        scoring='r2',
                        cv=folds,
                        verbose=1,
                        return_train_score=True)

model_cv.fit(X_train,y_train)


# In[123]:


cv_result_r = pd.DataFrame(model_cv.cv_results_)
cv_result_r['param_alpha'] = cv_result_r['param_alpha'].astype('float32')
cv_result_r


# In[124]:


plt.figure(figsize=(16,8))
plt.plot(cv_result_r['param_alpha'],cv_result_r['mean_train_score'])
plt.plot(cv_result_r['param_alpha'],cv_result_r['mean_test_score'])
plt.xlabel('Alpha')

plt.ylabel('R2 Score')
plt.show()


# In[125]:


# Checking the best parameter(Alpha value)
model_cv.best_params_


# In[126]:


ridge = Ridge(alpha = 20)
ridge.fit(X_train,y_train)

y_pred_train = ridge.predict(X_train)
print((r2_score(y_train,y_pred_train)))

y_pred_test = ridge.predict(X_test)
print((r2_score(y_test,y_pred_test)))


# In[129]:


model_parameter = list(ridge.coef_)
model_parameter.insert(0,ridge.intercept_)
cols = X_train.columns
cols=cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter)))
ridge_coef.columns = ['Feature','Coefficient']


# In[130]:


ridge_coef['Coefficient']=ridge_coef['Coefficient'].apply(lambda x :abs(x))


# In[131]:


ridge_coef.sort_values(by='Coefficient',ascending=False).head(10)


# # Subjective Questions

# ## Question 1
# 
# What is the optimal value of alpha for ridge and lasso regression? What will be the changes in the model if you choose double the value of alpha for both ridge and lasso? What will be the most important predictor variables after the change is implemented?

# ## Answer

# From above we can conclude that Lasso regression is better than Ridge regression as it also does feature elimination and makes the overall model simpler.

# ### Doubling the alpha and evaluating

# #### Lasso Regression

# In[132]:


#building the model
lasso = Lasso(alpha=0.002)
lasso.fit(X_train,y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print((r2_score(y_true=y_train,y_pred=y_train_pred)))
print((r2_score(y_true=y_test,y_pred=y_test_pred)))


# In[133]:


# lasso model parameters
model_parameters = list(lasso.coef_)
model_parameters.insert(0, lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[134]:


model_param = list(lasso.coef_)
model_param.insert(0,lasso.intercept_)
cols = X_train.columns
cols = cols.insert(0,'const')
lasso_coef = pd.DataFrame(list(zip(cols,model_param)))
lasso_coef.columns = ['Feature','Coefficient']


# In[135]:


lasso_coef['Coefficient']=lasso_coef['Coefficient'].apply(lambda x :abs(x))


# In[136]:


#Extracting top 10 coefficient
lasso_coef.sort_values(by='Coefficient',ascending=False).head(10)


# #### Ridge Regression 

# In[137]:


#printing the r2score for train and test
ridge = Ridge(alpha = 40)
ridge.fit(X_train,y_train)

y_pred_train = ridge.predict(X_train)
print((r2_score(y_train,y_pred_train)))

y_pred_test = ridge.predict(X_test)
print((r2_score(y_test,y_pred_test)))


# In[138]:


#checking the model parameters
model_parameter = list(ridge.coef_)
model_parameter.insert(0,ridge.intercept_)
cols = X_train.columns
cols = cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter)))
ridge_coef.columns = ['Feature','Coefficient']


# In[139]:


ridge_coef['Coefficient']=ridge_coef['Coefficient'].apply(lambda x :abs(x))


# In[140]:


#printing the coefficient
ridge_coef.sort_values(by='Coefficient',ascending=False).head(10)


# ## Question 3
# 
# After building the model, you realised that the five most important predictor variables in the lasso model are not available in the incoming data. You will now have to create another model excluding the five most important predictor variables. Which are the five most important predictor variables now?

# ## Answer

# In[141]:


X_train.drop('Neighborhood_StoneBr',axis=1,inplace=True)
X_train.drop('OverallQual',axis=1,inplace=True)
X_train.drop('Neighborhood_Crawfor',axis=1,inplace=True)
X_train.drop('Exterior1st_BrkFace',axis=1,inplace=True)
X_train.drop('Neighborhood_NridgHt',axis=1,inplace=True)
X_test.drop('Neighborhood_StoneBr',axis=1,inplace=True)
X_test.drop('OverallQual',axis=1,inplace=True)
X_test.drop('Neighborhood_Crawfor',axis=1,inplace=True)
X_test.drop('Exterior1st_BrkFace',axis=1,inplace=True)
X_test.drop('Neighborhood_NridgHt',axis=1,inplace=True)


# In[142]:


# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# cross validation
folds = 5

lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'r2', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train,y_train) 


# In[143]:


#check result of lasso regression
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[144]:


#graph for r2 score
plt.figure(figsize=(16,8))
plt.plot(cv_results['param_alpha'],cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'],cv_results['mean_test_score'])
plt.xscale('log')
plt.ylabel('R2 Score')
plt.xlabel('Alpha')
plt.show()


# In[145]:


#check for the best param
model_cv.best_params_


# In[146]:


#building the model
lasso = Lasso(alpha=0.0001)
lasso.fit(X_train,y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)

print((r2_score(y_true=y_train,y_pred=y_train_pred)))
print((r2_score(y_true=y_test,y_pred=y_test_pred)))


# In[147]:


# lasso model parameters
model_parameters = list(lasso.coef_)
model_parameters.insert(0, lasso.intercept_)
model_parameters = [round(x, 3) for x in model_parameters]
cols = X.columns
cols = cols.insert(0, "constant")
list(zip(cols, model_parameters))


# In[148]:


model_param = list(lasso.coef_)
model_param.insert(0,lasso.intercept_)
cols = X_train.columns
cols = cols.insert(0,'const')
lasso_coef = pd.DataFrame(list(zip(cols,model_param)))
lasso_coef.columns = ['Feature','Coefficient']


# In[149]:


lasso_coef['Coefficient']=lasso_coef['Coefficient'].apply(lambda x:abs(x))


# In[150]:


#Extracting top 10 coefficient
lasso_coef.sort_values(by='Coefficient',ascending=False).head(6)


# Above are the top 5 most important predictor variable
