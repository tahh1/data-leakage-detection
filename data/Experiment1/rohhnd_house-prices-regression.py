#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# # DATA

# In[5]:


train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")


# In[6]:


print(("Training data shape: ", train_data.shape))
print(("Testing data shape: ", test_data.shape))


# In[7]:


all_data = train_data.drop('SalePrice',axis=1).append(test_data)
all_data.shape


# In[8]:


all_data['GarageYrBlt'].dtype


# In[9]:


all_data['LotFrontage'] = all_data['LotFrontage'].astype('float64')


# In[10]:


numerical_columns = []
object_columns = []
for name, typed in zip(all_data.columns, all_data.dtypes):
    if typed == 'object':
        object_columns.append(name)
    elif typed == 'int64' or typed == 'float64':
        numerical_columns.append(name)
    print((name, ' - ',typed))
    
extra_cat_columns = ['MSSubClass','GarageYrBlt', 'MoSold', 'YrSold']
for name in extra_cat_columns:
    object_columns.append(name)


# In[11]:


for name in object_columns:
    #print(all_data[name].dtypes)
    if all_data[name].dtypes == 'object':
        all_data[name] = all_data[name].fillna('NaN')
    elif all_data[name].dtypes == 'int64':
        all_data[name] = all_data[name].fillna(999)
    all_data[name] = label_encoder.fit_transform(all_data[name])
#all_data.head()


# ## Imputing missing values

# In[12]:


missing_value_columns = []
for a,x in list(all_data.isna().sum().items()):
    if x>0:
        print((a,x))
        missing_value_columns.append(a)


# In[13]:


imputer = IterativeImputer(missing_values=np.nan,max_iter=10, initial_strategy='median', n_nearest_features=4)
imputer_model = imputer.fit(all_data)


# In[14]:


imputed_data = pd.DataFrame(data = imputer_model.transform(all_data), columns=list(all_data.columns))


# In[15]:


for a,x in list(imputed_data.isna().sum().items()):
    if x>0:
        print((a,x))


# In[16]:


imputed_data.head()


# ## Final Training and Test Data

# In[17]:


final_train_data = imputed_data[:1460]
final_train_data = final_train_data.merge(train_data[['Id','SalePrice']], on ='Id', how='left')
final_train_data.head()


# In[18]:


final_test_data = pd.DataFrame(data=imputed_data[1460:2919], columns=list(all_data.columns))


# Scaling data

# In[19]:


scaled_train_data = scaler.fit_transform(final_train_data.drop(['Id','SalePrice'], axis=1))
x_train, x_val, y_train, y_val = train_test_split(scaled_train_data, final_train_data['SalePrice'], test_size=0.25, random_state=42)


# In[20]:


print((len(x_train), len(y_train), len(x_val), len(y_val)))


# ## Linear Regression

# In[18]:


lr = LinearRegression(normalize=True)
model = lr.fit(x_train, y_train)


# In[19]:


lr_prediction = model.predict(x_val)


# In[20]:


model.score(x_train,y_train)


# In[21]:


plt.scatter(y_val, lr_prediction)
plt.show()


# In[22]:


print(("Linear regression rmse score: ", np.sqrt(mean_squared_error(y_val, lr_prediction))))


# ## Lasso Regression

# In[23]:


lasso = Lasso()
paramgrid = {
    'alpha':list(np.arange(0.85,1.01,0.01)),
    'tol': list(np.arange(0,1.01, 0.01))
}


# In[24]:


gscv = GridSearchCV(estimator=lasso, param_grid=paramgrid, cv= 5, n_jobs=-1, verbose=1)
gscv.fit(x_train, y_train)


# In[25]:


gscv.best_params_


# In[26]:


lasso = Lasso(alpha=1.01, tol=0.15)
model = lasso.fit(x_train, y_train)


# In[27]:


model.score(x_train, y_train)


# In[28]:


lasso_prediction = model.predict(x_val)
print(("Lasso regression rmse score: ", np.sqrt(mean_squared_error(y_val, lasso_prediction))))


# ## Ridge Regression

# In[29]:


ridge = Ridge()
paramgrid = {
    'alpha':[0.1,0.3,0.5,0.7,1],
    'tol': [0.1,0.3,0.5,0.7,1]
}


# In[30]:


gscv = GridSearchCV(estimator=ridge, param_grid=paramgrid, cv= 5, n_jobs=-1, verbose=1)


# In[31]:


gscv.fit(x_train, y_train)


# In[32]:


gscv.best_params_


# In[33]:


ridge = Ridge(alpha=1, tol=0.1, solver='cholesky')
model = ridge.fit(x_train, y_train)


# In[34]:


model.score(x_train, y_train)


# In[35]:


ridge_prediction = model.predict(x_val)


# In[36]:


print(("Ridge regression rmse score: ", np.sqrt(mean_squared_error(y_val, ridge_prediction))))


# ## XGBoost

# In[39]:


xgb = XGBRegressor()
paramgrid = {'learning_rate':[0.1,0.3],
            'max_depth': [5,11,20],
            'min_child_weight': [7,11],
            'subsample':[0.1,0.3,0.5],
            'colsample_bytree':[0.3,0.5],
            'n_estimators':[100,200],
            'objective':['reg:squarederror']}


# In[40]:


gscv = GridSearchCV(estimator=xgb, param_grid=paramgrid, cv= 5, n_jobs=-1, verbose=1)


# In[41]:


gscv.fit(x_train, y_train)


# In[42]:


gscv.best_params_


# In[43]:


xgb = XGBRegressor(learning_rate=0.1, colsample_bytree=0.3, max_depth=9, min_child_weight=11, n_estimators=100, objective = 'reg:squarederror', subsample=0.3)
model = xgb.fit(x_train, y_train)


# In[44]:


model.score(x_train,y_train)


# In[45]:


xgb_prediction = model.predict(x_val)


# In[46]:


print(("XGBoost rmse score: ", np.sqrt(mean_squared_error(y_val, xgb_prediction))))


# ## XGBoost RF Regression

# In[47]:


xgbrf = XGBRFRegressor()
model = xgbrf.fit(x_train, y_train)


# In[48]:


model.score(x_train, y_train)


# In[49]:


xgbrf_prediction = model.predict(x_val)


# In[50]:


print(("XGBoostRF rmse score: ", np.sqrt(mean_squared_error(y_val, xgbrf_prediction))))


# ## DecisionTree

# In[51]:


dtr = DecisionTreeRegressor()
paramgrid = {'max_depth':[5,11,15],
             'min_samples_leaf':[20,40,80],
             'max_features':[20,40,60],
             'max_leaf_nodes':[40,60,80]
            }


# In[52]:


gscv = GridSearchCV(estimator=dtr, param_grid=paramgrid, cv= 5, n_jobs=-1, verbose=1)


# In[53]:


gscv.fit(x_train, y_train)


# In[54]:


gscv.best_params_


# In[58]:


dtr = DecisionTreeRegressor(max_depth=15, max_features=60, max_leaf_nodes=40, min_samples_leaf=20)
model = dtr.fit(x_train, y_train)


# In[59]:


model.score(x_train, y_train)


# In[60]:


dt_prediction = model.predict(x_val)
print(("Decision Tree rmse score: ", np.sqrt(mean_squared_error(y_val, dt_prediction))))


# ## Random Forest

# In[22]:


rf = RandomForestRegressor()
paramgrid = {'max_depth':[7,11,15],
             'min_samples_leaf':[5,20,60],
             'max_leaf_nodes':[5,20,60],
            'n_estimators':[50,100,500]}


# In[23]:


gscv = GridSearchCV(estimator=rf, param_grid=paramgrid, cv= 5, n_jobs=-1, verbose=1)
gscv.fit(x_train, y_train)


# In[24]:


gscv.best_params_


# In[25]:


rf = RandomForestRegressor(max_depth=11, max_features=20, max_leaf_nodes=60, min_samples_leaf=5, min_samples_split=5, n_estimators=100)
model = rf.fit(x_train, y_train)


# In[26]:


model.score(x_train, y_train)


# In[27]:


rf_prediction = model.predict(x_val)
print(("Decision Tree rmse score: ", np.sqrt(mean_squared_error(y_val, rf_prediction))))


# ## AdaBoost

# Adaboost on XGBoost

# In[28]:


ada_booster = AdaBoostRegressor(XGBRegressor(learning_rate=0.1, colsample_bytree=0.3, max_depth=9, min_child_weight=11, n_estimators=100, objective = 'reg:squarederror', subsample=0.3), n_estimators=1200)
model = ada_booster.fit(scaled_train_data, final_train_data['SalePrice'])


# In[29]:


model.score(scaled_train_data, final_train_data['SalePrice'])


# In[30]:


ada_prediction = model.predict(x_val)
print(("AdaBoost XGBoost rmse score: ", np.sqrt(mean_squared_error(y_val, ada_prediction))))


# AdaBoost on DecisionTree

# In[31]:


ada_booster = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15, max_features=40, max_leaf_nodes=40, min_samples_leaf=20), n_estimators=1200)
model = ada_booster.fit(x_train,y_train)


# In[32]:


model.score(x_train, y_train)


# In[33]:


ada_prediction = model.predict(x_val)
print(("AdaBoost DecisionTree rmse score: ", np.sqrt(mean_squared_error(y_val, ada_prediction))))


# Adaboost on XGBRF

# In[34]:


ada_booster = AdaBoostRegressor(XGBRFRegressor(), n_estimators=1000)
model = ada_booster.fit(x_train,y_train)


# In[35]:


model.score(x_train, y_train)


# In[36]:


ada_prediction = model.predict(x_val)
print(("AdaBoost XGBRF rmse score: ", np.sqrt(mean_squared_error(y_val, ada_prediction))))


# AdaBoost on Random Forest

# In[37]:


ada_booster = AdaBoostRegressor(RandomForestRegressor(max_depth=11, max_features=20, max_leaf_nodes=60, min_samples_leaf=5, min_samples_split=5, n_estimators=100), n_estimators=1000)
model = ada_booster.fit(x_train,y_train)


# In[38]:


model.score(x_train, y_train)


# In[39]:


ada_prediction = model.predict(x_val)
print(("AdaBoost XGBRF rmse score: ", np.sqrt(mean_squared_error(y_val, ada_prediction))))


# ## Stacking
# 

# In[40]:


from sklearn.ensemble import StackingRegressor


# In[41]:


stack = [
    ('lasso', Lasso(alpha=1.01, tol=0.15)),
    ('rf',RandomForestRegressor(max_depth=11, max_features=20, max_leaf_nodes=60, min_samples_leaf=5, min_samples_split=5, n_estimators=100)),
    ('xgb', XGBRegressor(learning_rate=0.1, colsample_bytree=0.3, max_depth=9, min_child_weight=11, n_estimators=100, objective = 'reg:squarederror', subsample=0.3))
]

stacked = StackingRegressor(estimators=stack,
                           cv=5)


# In[42]:


stacked_model = stacked.fit(x_train,y_train)
stacked_model.score(x_train, y_train)


# In[43]:


ada = AdaBoostRegressor(stacked, n_estimators=200)
ada.fit(x_train, y_train)


# In[44]:


ada.score(x_train, y_train)


# ## Submissions

# In[45]:


submission.head()


# In[ ]:


y_pred = ada.predict(scaler.transform(final_test_data.drop('Id', axis=1)))
y_pred


# In[ ]:


submission['SalePrice'] = y_pred


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




