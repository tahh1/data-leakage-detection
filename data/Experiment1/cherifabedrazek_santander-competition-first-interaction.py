#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# In[2]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
sample_submission_df = pd.read_csv('../input/sample_submission.csv')


# ### Rescaling Features

# Example of feature scale of the first 3 vars

# In[3]:


scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(train_df[['var_0','var_1','var_2']])
scaled_df = pd.DataFrame(scaled_df, columns=['var_0', 'var_1', 'var_2'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(train_df['var_0'], ax=ax1)
sns.kdeplot(train_df['var_1'], ax=ax1)
sns.kdeplot(train_df['var_2'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['var_0'], ax=ax2)
sns.kdeplot(scaled_df['var_1'], ax=ax2)
sns.kdeplot(scaled_df['var_2'], ax=ax2)
plt.show()


# #### Now we scale all test and train data

# In[4]:


columns = train_df.drop(['ID_code','target'],axis=1).columns
scaler = preprocessing.StandardScaler()
## train scale

scaled_train_df = scaler.fit_transform(train_df[columns])
scaled_train_df = pd.DataFrame(scaled_train_df, columns=columns)

##test scale
scaled_test_df = scaler.fit_transform(test_df.drop(['ID_code'],axis=1))
scaled_test_df = pd.DataFrame(scaled_test_df, columns=columns)


# #### Check on random variables

# In[5]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(train_df['var_10'], ax=ax1)
sns.kdeplot(train_df['var_20'], ax=ax1)
sns.kdeplot(train_df['var_22'], ax=ax1)
ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_train_df['var_10'], ax=ax2)
sns.kdeplot(scaled_train_df['var_20'], ax=ax2)
sns.kdeplot(scaled_train_df['var_22'], ax=ax2)
plt.show()


# Adding the target and ID_code

# In[6]:


scaled_train_df['ID_code'] = train_df['ID_code']
scaled_test_df['ID_code'] = test_df['ID_code']
scaled_train_df['target'] = train_df['target']


# ## Feature engineering

# I will start by extracting non negative variables

# In[7]:


seriii = (train_df[columns].describe().loc['min']>=0)
positive_vars = seriii[seriii==True].index.tolist() ## list of positive vars


# Non negative variables could be time or money spent, so will start by generating the square of each variable

# In[8]:


for column in positive_vars:
    scaled_train_df[column+'_squared'] = scaled_train_df[column]**2
    scaled_test_df[column+'_squared'] = scaled_test_df[column]**2


# We can also add other features based on the original data non negative feature where we take the fractional part as it may be important in some cases

# In[9]:


for column in positive_vars:
    scaled_train_df[column+'_fraction'] = train_df[column] - (train_df[column]).astype(int)
    scaled_test_df[column+'_fraction'] = test_df[column] - (test_df[column]).astype(int)


# ### Let's have a look at feature importance

# In[10]:


y = scaled_train_df.target
X = scaled_train_df.drop(['target','ID_code'], axis=1)


# #### Feature importance

# In[11]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)

#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# #### Heatmap correlation

# In[12]:


corrmat = scaled_train_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(scaled_train_df[top_corr_features].corr(),annot=False,cmap="RdYlGn")


# #### Univariate Selection

# In[13]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(train_df[positive_vars],y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(train_df[positive_vars].columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print((featureScores.nlargest(10,'Score')))  #print 10 best features


# ## Lets try to build a model right now to check our score

# ### MLP classifier

# In[14]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=27)


# In[15]:


clf = MLPClassifier(hidden_layer_sizes=(400,200,200,100), max_iter=200, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)


# In[16]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# In[17]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[18]:


X_submission = scaled_test_df.drop('ID_code',axis=1)


# In[19]:


sub_preds = clf.predict(X_submission)


# In[20]:


clf_submsission = pd.DataFrame(test_df.ID_code)
clf_submsission['target'] = sub_preds
clf_submsission.to_csv('clfv1.csv',index=False)


# ### DNN with Tensorflow

# In[21]:


import tensorflow as tf
np.random.seed(1337)


# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train  = y_train.astype(int)
y_test  = y_test.astype(int)
batch_size =len(X)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)


# In[23]:


## resclae
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Train
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# test
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))


# In[24]:


feature_columns = [tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]


# In[25]:


estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[300, 100], 
    n_classes=2,
    model_dir = '/train/DNN')


# ### I got error while training 
# #### Train the estimator
# train_input = tf.estimator.inputs.numpy_input_fn(
#     x={"x": X_train_scaled},
#     y=y_train,
#     batch_size=50,
#     shuffle=False,
#     num_epochs=None)
# estimator.train(input_fn = train_input,steps=1000) 
# eval_input = tf.estimator.inputs.numpy_input_fn(
#     x={"x": X_test_scaled},
#     y=y_test, 
#     shuffle=False,
#     batch_size=X_test_scaled.shape[0],
#     num_epochs=1)
# estimator.evaluate(eval_input,steps=None) 

# ### XGboost model

# In[26]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# In[27]:


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[28]:


model = XGBClassifier(learning_rate=0.01,n_estimators=300,n_jobs=4,seed=223,max_depth=10,subsample=0.7,min_child_weight=5)
model.fit(X_train, y_train)


# In[29]:


y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[30]:


accuracy = accuracy_score(y_test, y_pred)
print(("Accuracy: %.2f%%" % (accuracy * 100.0)))


# In[31]:


X_submission = scaled_test_df.drop('ID_code',axis=1)


# In[32]:


y_pred_submission = model.predict(X_submission)
submission_predictions = [round(value) for value in y_pred_submission]


# In[33]:


XGB_submsission = pd.DataFrame(test_df.ID_code)
XGB_submsission['target'] = y_pred_submission


# In[34]:


XGB_submsission.to_csv('XGB_submsission_v01.csv',index=False)


# ### LGBM classifier

# In[35]:


from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : 42,
    "verbosity" : 1,
    "seed": 42
}

folds = StratifiedKFold(n_splits=10)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_submission.shape[0])
for fold_, (trn_, val_) in enumerate(folds.split(X, y)):
    trn_x, trn_y = X.iloc[trn_], y[trn_]
    val_x, val_y = X.iloc[val_], y[val_]

    model = lgb.LGBMRegressor(**params, n_estimators=100000)
    model.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=3000, verbose=1000)

    oof_preds[val_] = model.predict(val_x, num_iteration=model.best_iteration_)
    sub_preds += model.predict(X_submission, num_iteration=model.best_iteration_) / folds.n_splits


# In[36]:


sub_preds = model.predict(X_submission)


# Tune thresh function

# In[37]:


def thresh(x):
    return (x>0.15).astype(int)


# In[38]:


submission_3_predictions = [thresh(value) for value in sub_preds]


# Submission 2 with round

# In[39]:


submission_2_predictions = [round(value) for value in sub_preds]


# In[40]:


LGBM_submsission = pd.DataFrame(test_df.ID_code)
LGBM_submsission['target'] = submission_2_predictions


# Sumbission 3 with thresh

# In[41]:


LGBM_submsission = pd.DataFrame(test_df.ID_code)
LGBM_submsission['target'] = submission_3_predictions
LGBM_submsission.to_csv('LGBM_submsission_v1.csv',index=False)


# Saving data frame with real value data

# In[42]:


LGBM_submsission = pd.DataFrame(test_df.ID_code)
LGBM_submsission['target'] = sub_preds
LGBM_submsission.to_csv('real_value_output.csv',index=False)

