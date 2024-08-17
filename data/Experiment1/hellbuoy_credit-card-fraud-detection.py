#!/usr/bin/env python
# coding: utf-8

# # Capstone Project: Credit Card Fraud Detection

# ## Problem Statement
# 
# The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.
# 
# In this project, you will analyse customer-level data which has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 
# 
# The dataset is taken from the Kaggle website and it has a total of 2,84,807 transactions, out of which 492 are fraudulent. Since the dataset is highly imbalanced, so it needs to be handled before model building.

# ![](https://www.xenonstack.com/wp-content/uploads/xenonstack-credit-card-fraud-detection.png)

# ## Business Problem Overview
# 
# For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.
# 
# It has been estimated by Nilson report that by 2020 the banking frauds would account to $30 billion worldwide. With the rise in digital payment channels, the number of fraudulent transactions is also increasing with new and different ways. 
# 
# In the banking industry, credit card fraud detection using machine learning is not just a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees, and denials of legitimate transactions.

# ## Project Pipeline
# The project pipeline can be briefly summarized in the following four steps:
# 
# **Data Understanding:** Here, you need to load the data and understand the features present in it. This would help you choose the features that you will need for your final model.<br>
# 
# **Exploratory data analytics (EDA):** Normally, in this step, you need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. For the current data set, because Gaussian variables are used, you do not need to perform Z-scaling. However, you can check if there is any skewness in the data and try to mitigate it, as it might cause problems during the model-building phase.<br>
# 
# **Train/Test Split:** Now you are familiar with the train/test split, which you can perform in order to check the performance of your models with unseen data. Here, for validation, you can use the k-fold cross-validation method. You need to choose an appropriate k value so that the minority class is correctly represented in the test folds.<br>
# 
# **Model-Building/Hyperparameter Tuning:** This is the final step at which you can try different models and fine-tune their hyperparameters until you get the desired level of performance on the given dataset. You should try and see if you get a better model by the various sampling techniques.<br>
# 
# **Model Evaluation:** Evaluate the models using appropriate evaluation metrics. Note that since the data is imbalanced it is is more important to identify which are fraudulent transactions accurately than the non-fraudulent. Choose an appropriate evaluation metric which reflects this business goal.

# #### This kernel is based on the assignment by IIITB collaborated with upgrad.

# #### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated.

# #### Below are the steps which we will be basically following:
# 
# 1. [Step 1: Reading and Understanding the Data](#1)
# 1. [Step 2: Data Cleansing](#2)
#    - Null Check
#    - Data type check
# 1. [Step 3: Data Visualization](#3)
#    - Imbalanced Data check
#    - Data Distribution Plots
# 1. [Step 4: Splitting the Data into Training and Testing Sets](#4)
#    - Stratified Train Test Split
#    - Skewness Check and Fix
# 1. [Step 5:  Building a Logistic Model](#5)
#    - Logistic Regression on Imbalanced Data
#    - Random Forest on Imbalanced Data
#    - XGBoost on Imbalanced Data
#    - Logistic Regression on Balanced Data (Random Over Sampling,SMOTE,ADASYN)
#    - Random Forest on Balanced Data (Random Over Sampling,SMOTE,ADASYN)
#    - XGBoost on Balanced Data (Random Over Sampling,SMOTE,ADASYN)
#    - Stratified K-Fold and Hyperparameter Tuning
#    - Classification Report
#    - Confusion Matrix
#    - ROC AUC Curve
# 1. [Step 6: Final Analysis](#6)
#    - Best Model Selection
#    - Features Importance
# 1. [Step 7: Closing Statement](#7)
# 

# <a id='1'></a><br>
# ## Step 1: Reading and Understanding the Data

# In[1]:


# import all the required libraries and dependencies for dataframe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')
import matplotlib.ticker as plticker
get_ipython().run_line_magic('matplotlib', 'inline')

# import all the required libraries and dependencies for machine learning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import statsmodels.api as sm
import pickle
import gc 
from sklearn import svm
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


# Local file path.Please change the path accordingly.

path = '../input/credit-card-fraud/creditcard.csv'


# In[3]:


# Reading the Credit Card file on which analysis needs to be done

df_card = pd.read_csv(path)
df_card.head()


# In[4]:


# Shape of the Credit card dataframe

df_card.shape


# #### Inference:
# We got `284807` records and `31` columns in our dataset.

# In[5]:


# Data Description

df_card.describe()


# In[6]:


# Data Information

df_card.info()


# <a id='2'></a><br>
# ## Step 2 : Data Cleansing

# We need to perform basic cleansing check in order to feed correct data to the model.

# In[7]:


# Calculating the Missing Value% in the DF

df_null = df_card.isnull().mean()*100
df_null.sort_values(ascending=False).head()


# We observed in the dataset has no null values and Hence, no Null treatment is required.

# In[8]:


# Datatype check for the dataframe

df_card.dtypes


# #### Inference:
# None of the columns have inconsistent datatype.Hence, no conversion is required.

# <a id='3'></a><br>
# ## Step 3 : Data Visualization

# #### Visualizing the Distribution of `Fraudulent` Variable

# In[9]:


plt.figure(figsize=(13,7))
plt.subplot(121)
plt.title('Fraudulent BarPlot', fontweight='bold',fontsize=14)
ax = df_card['Class'].value_counts().plot(kind='bar')
total = float(len(df_card))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.5f}'.format(height/total),
            ha="center") 


plt.subplot(122)
df_card["Class"].value_counts().plot.pie(autopct = "%1.5f%%")
plt.show()


# In[10]:


classes=df_card['Class'].value_counts()
normal_share=classes[0]/df_card['Class'].count()*100
fraud_share=classes[1]/df_card['Class'].count()*100
print(normal_share)
print(fraud_share)


# In this scenario accuracy score to evaluate our classification algorithm will not be correct.Just using accuracy as the evaluation metric will predit every case as `0` Non Fraud and hence it would be wrong.

# In[11]:


# Box Plot of amount for both classes
plt.figure(figsize = (7, 6))
a=sns.boxplot(x = 'Class', y = 'Amount',hue='Class', data = df_card,showfliers=False) 
plt.setp(a.get_xticklabels(), rotation=45)


# In[12]:


# KDE plot to visualize the distribution of Amount for both the classes
plt.rcParams['figure.figsize'] = [10,6]
sns.kdeplot(df_card.loc[df_card['Class'] == 0, 'Amount'], label = 'Non Fraud')
sns.kdeplot(df_card.loc[df_card['Class'] == 1, 'Amount'], label = 'Fraud')
plt.title('Distribution of Amount by Target Value')
plt.xlabel('Amount')
plt.ylabel('Density')


# #### Inference:
# 
# - The distribution of amount for Fraud transactions is much higher than non-fradualent transactions. 

# In[13]:


# Time Distribution plot for transactions 
plt.figure(figsize=(15,7))

plt.title('Distribution of Transaction Time')
sns.distplot(df_card['Time'].values/(60*60))


# In[14]:


# Storing Fraud and non-Fraud transactions 

df_nonfraud = df_card[df_card.Class == 0]
df_fraud = df_card[df_card.Class == 1]


# In[15]:


#Scatter plot between Time and Amount

fig = plt.figure(figsize = (8,8))
plt.scatter(df_nonfraud.Amount, df_nonfraud.Time.values/(60*60),alpha=0.5,label='Non Fraud')
plt.scatter(df_fraud.Amount, df_fraud.Time.values/(60*60),alpha=1,label='Fraud')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.title('Scatter plot between Amount and Time ')
plt.show()


# In[16]:


# Plot of high value transactions($200-$2000)

bins = np.linspace(200, 2000, 100)
plt.hist(df_nonfraud.Amount, bins, alpha=1, density=True, label='Non-Fraud')
plt.hist(df_fraud.Amount, bins, alpha=1, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions \$200-$2000)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)")
plt.show()


# #### Inference:
# 
# - In the long tail, fraud transaction happened more frequently.
# 
# - It seems It would be hard to differentiate fraud from normal transactions by transaction amount alone.

# In[17]:


# Plot of transactions in 48 hours

bins = np.linspace(0, 48, 48)
plt.hist((df_nonfraud.Time/(60*60)), bins, alpha=1,label='Non-Fraud')
plt.hist((df_fraud.Time/(60*60)), bins, alpha=0.6,label='Fraud')
plt.legend(loc='upper right')
plt.title("Percentage of transactions by hour")
plt.xlabel("Transaction time from first transaction in the dataset (hours)")
plt.ylabel("Percentage of transactions (%)")
plt.show()


# #### Inference:
# Hour `zero` corresponds to the hour the first transaction happened and not necessarily `12-1 AM`. Given the heavy decrease in normal transactions from hours `1` to `8` and again roughly at hours `24` to `32`, 
# it seems fraud tends to occur at higher rates during the night.

# <a id='4'></a><br>
# ## Step 4: Splitting the Data into Training and Testing Sets
# 
# As we know, the first basic step for regression is performing a train-test split.

# In[18]:


# Putting the feature variable into X

X = df_card.drop(['Class'],axis = 1)
X.head(2)


# In[19]:


# Putting the Target variable to y

y = df_card['Class']


# In[20]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[21]:


# Splitting the data into Train and Test set
kfold = 4
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.3, random_state=9487)
for train_index, test_index in sss.split(X, y):
        print(("TRAIN:", train_index, "TEST:", test_index))
        X_train, X_test = X.iloc, X.iloc
        y_train, y_test = y[train_index], y[test_index]


# In[22]:


# Checking Skewness of data

plt.rcParams['figure.figsize'] = [10,8]
plt.hist(df_card['Amount'],edgecolor='k',bins = 5)
plt.title('Transaction Amount')
plt.xlabel('Amount in USD') 
plt.ylabel('Count')


# In[23]:


from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer


# In[24]:


pt = preprocessing.PowerTransformer(copy=False)
PWTR_X = pt.fit_transform(X)


# <a id='5'></a><br>
# ## Step 5: Building a Logistic Model

# Here,instead of `Accuracy` we are very much interested in the recall score, because that is the metric that will help us try to capture the most fraudulent transactions.
# Increase of Recall comes at a price of Precision. In this case predicting a transaction fradulant which actually is not is not a big concern.

# ## Imbalanced Data Set

# ### Logistic Regression

# #### Model 1 : Logistic Regression on Imbalanced Data

# In[25]:


# Splitting dataset into test and train sets in 70:30 ratio after applying Power Transform

kfold = 4
sss = StratifiedShuffleSplit(n_splits=kfold, test_size=0.3, random_state=9487)
for train_index, test_index in sss.split(PWTR_X, y):
        print(("TRAIN:", train_index, "TEST:", test_index))
        X_train, X_test = PWTR_X[train_index], PWTR_X[test_index]
        y_train, y_test = y[train_index], y[test_index]


# In[26]:


from sklearn.linear_model import LogisticRegression

# Fit a logistic regression model to train data
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)


# In[27]:


# Predict on test data
y_predicted = model_lr.predict(X_test)


# In[28]:


# Evaluation Metrics

print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[29]:


# Function for roc_curve
def plot_roc_curve(fpr,tpr,roc_auc):
    plt.plot(fpr, tpr, linewidth=5, label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='upper right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[30]:


# tpr and fpr
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)


# In[31]:


# Plotting the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.86
# - Recall : 0.60
# - F1-score : 0.70
# - Accuracy : 0.85
# - ROC AUC : 0.80

# #### Hyperparameter Tuning Logisitic Regression

# #### Model 2 : Logistic Regression on Imbalanced Data with K-Fold and Hypertuning

# In[32]:


from imblearn.metrics import sensitivity_specificity_support


# In[33]:


# Number of folds

n_folds = 5
# parameters 
params ={'C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'penalty': ['l1', 'l2']}

lrh = LogisticRegression()

model_lrh = GridSearchCV(estimator=lrh, cv=n_folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)


# In[34]:


# Fitting the model

model_lrh.fit(X_train,y_train)


# In[35]:


pd.DataFrame(model_lrh.cv_results_)


# In[36]:


print(("Logistic Regression with PCA Best AUC : ", model_lrh.best_score_))
print(("Logistic Regression with PCA Best hyperparameters: ", model_lrh.best_params_))


# In[37]:


# Passing the best parameteres
model_lrh_tuned = LogisticRegression(penalty='l2',C=0.1)


# In[38]:


# Predicting on test data

model_lrh_tuned.fit(X_train,y_train)
y_predicted = model_lrh_tuned.predict(X_test)


# In[39]:


#Evaluation Metrices

print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[40]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.85
# - Recall : 0.59
# - F1-score : 0.70
# - Accuracy : 0.99
# - ROC AUC : 0.79

# ### Random Forest

# #### Model 3 : Random Forest on Imbalanced Data

# In[41]:


#Initializing Random forest and creating model

from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier(n_jobs=-1, 
                             random_state=2018,
                             criterion='gini',
                             n_estimators=100,
                             verbose=False)


# In[42]:


# Fitting the model on Train data and Predicting on Test data

model_rfc.fit(X_train,y_train)
y_predicted = model_rfc.predict(X_test)


# In[43]:


# Evaluation Metrics

print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[44]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.94
# - Recall : 0.70
# - F1-score : 0.80
# - Accuracy : 0.99
# - ROC AUC : 0.85

# #### Hyperparameter Tuning Random Forest

# #### Model 4 : Random Forest on Imbalanced Data with K-Fold and Hyperparamater Tuning

# In[45]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


# In[46]:


# Defining Parameters
params = { 
    'n_estimators': [200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[47]:


# Stratified K Fold
cross_val = StratifiedKFold(n_splits=3)
index_iterator = cross_val.split(X_train, y_train)
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 50, cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[48]:


# Fitting the model on train data
# clf_random.fit(X_train, y_train)


# In[49]:


# Scores of RandomizedSearchCV
#scores = clf_random.cv_results_
#pd.DataFrame(scores).head()


# In[50]:


#print(clf_random.best_score_)
#print(clf_random.best_params_)
#print(clf_random.best_index_)


# In[51]:


# Passing the best parameteres based on Randomized Search CV
model_rfc_tuned = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud
                               criterion='gini',
                               max_depth=5,
                               max_features='sqrt',
                               min_samples_leaf=10,
                               n_estimators=200,
                               n_jobs=-1, 
                               random_state=5)


# In[52]:


# Fitting the model on Train data and Predicting on Test Data

model_rfc_tuned.fit(X_train,y_train)
y_predicted = model_rfc_tuned.predict(X_test)


# In[53]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[54]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.81
# - Recall : 0.71
# - F1-score : 0.76
# - Accuracy : 0.99
# - ROC AUC : 0.85

# ### XG Boost

# #### Model 5 : XG Boost on Imbalanced Data

# In[55]:


#Initializing Random forest and creating model
model_xgb = XGBClassifier()


# In[56]:


# Fitting the model on Train data and Predicting on Test data
model_xgb.fit(X_train,y_train)
y_predicted = model_xgb.predict(X_test)


# In[57]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[58]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.95
# - Recall : 0.74
# - F1-score : 0.83
# - Accuracy : 0.99
# - ROC AUC : 0.87

# #### Hyperparameter Tuning XGB

# Model 6 : XGB on Imbalanced Data with K-Fold and Hyperparamater Tuning

# In[59]:


# Defining parameters
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[60]:


# Stratified K Fold
cross_val = StratifiedKFold(n_splits=5)
index_iterator = cross_val.split(X_train, y_train)


xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 


xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[61]:


# Fitting the model on train data
#xgb_random.fit(X_train, y_train)


# In[62]:


# scores of RandomizedSearchCV
#scores = xgb_random.cv_results_
#pd.DataFrame(scores).head()


# In[63]:


#print(xgb_random.best_score_)
#print(xgb_random.best_params_)
#print(xgb_random.best_index_)


# In[64]:


# Passing the best parameteres based on Randomized Search CV
model_xgb_tuned = XGBClassifier(min_child_weight= 5,
        gamma= 1.5,
        subsample= 1.0,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[65]:


# Fitting the model on Train data and Predicting on Test data
model_xgb_tuned.fit(X_train,y_train)
y_predicted = model_xgb_tuned.predict(X_test)


# In[66]:


# Evaluation metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[67]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.95
# - Recall : 0.72
# - F1-score : 0.82
# - Accuracy : 0.99
# - ROC AUC : 0.85

# ## Balanced Data Set

# In[68]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# Resample training data
ros = RandomOverSampler()
smote = SMOTE(random_state=5)
adasyn = ADASYN(random_state=5)

X_train_ros, y_train_ros = ros.fit_sample(X_train,y_train)
X_train_smote, y_train_smote = smote.fit_sample(X_train,y_train)
X_train_adasyn, y_train_adasyn =adasyn.fit_sample(X_train,y_train)


# ### Logistic Regression

# #### Model 7 : Logistic Regression on ROS Balanced Data

# In[69]:


# Fit a logistic regression model to our data
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_ros, y_train_ros)

# Obtain model predictions
y_predicted = model_lr.predict(X_test)


# In[70]:


# Evaluation Metrics
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[71]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.06
# - Recall : 0.91
# - F1-score : 0.11
# - Accuracy : 0.97
# - ROC AUC : 0.94

# #### Model 8 : Logistic Regression on SMOTE Balanced Data

# In[72]:


# Fit a logistic regression model to our data
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_smote, y_train_smote)

# Obtain model predictions
y_predicted = model_lr.predict(X_test)


# In[73]:


# Evaluation Metrics
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# In[74]:


# Create true and false positive rates
fpr, tpr, threshold = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)
# Plot the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.06
# - Recall : 0.91
# - F1-score : 0.11
# - Accuracy : 0.97
# - ROC AUC : 0.93

# #### Model 9 : Logistic Regression on ADASYN Balanced Data

# In[75]:


# Fit a logistic regression model to our data
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train_adasyn, y_train_adasyn)

# Obtain model predictions
y_predicted = model_lr.predict(X_test)


# In[76]:


# Evaluation Metrics
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.02
# - Recall : 0.91
# - F1-score : 0.04
# - Accuracy : 0.91
# - ROC AUC : 0.91

# ### Random Forest

# #### Model 10 : Random Forest on ROS Balanced Data

# In[77]:


# Insantiate Model
model_rfc = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12}, # 0: non-fraud , 1:fraud
                               criterion='entropy',
                               max_depth=10, # Change depth of model
                               min_samples_leaf=10, # Change the number of samples in leaf nodes
                               n_estimators=20, # Change the number of trees to use
                               n_jobs=-1, 
                               random_state=5)


# In[78]:


# Fit the model on train data and predict on test data 
model_rfc.fit(X_train_ros,y_train_ros)
y_predicted = model_rfc.predict(X_test)


# In[79]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.46
# - Recall : 0.79
# - F1-score : 0.58
# - Accuracy : 0.99
# - ROC AUC : 0.89

# #### Model 11 : Random Forest on SMOTE Balanced Data

# In[80]:


# Fit the model on train data and predict on test data 
model_rfc.fit(X_train_smote,y_train_smote)
y_predicted = model_rfc.predict(X_test)


# In[81]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.06
# - Recall : 0.84
# - F1-score : 0.12
# - Accuracy : 0.97
# - ROC AUC : 0.91

# #### Model 12 : Random Forest on ADASYN Balanced Data

# In[82]:


# Fit the model on train data and predict on test data 
model_rfc.fit(X_train_adasyn,y_train_adasyn)
y_predicted = model_rfc.predict(X_test)


# In[83]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.03
# - Recall : 0.84
# - F1-score : 0.06
# - Accuracy : 0.95
# - ROC AUC : 0.89

# Let's do a Hyperparameter tuning on ROS data as it gave a better Recall and Precision

# #### Hyperparameter Tuning Random Forest on ROS Data

# #### Model 13 : Hyper Tuning  model Random Forest on ROS Balanced Data

# In[84]:


params = { 
    'n_estimators': [200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[85]:


cross_val = StratifiedKFold(n_splits=3)
index_iterator = cross_val.split(X_train_ros, y_train_ros)
clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = params, n_iter = 50, cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[86]:


#clf_random.fit(X_train_ros, y_train_ros)


# In[87]:


# Scores of RandomizedSearchCV
#scores = clf_random.cv_results_
#pd.DataFrame(scores).head()


# In[88]:


#print(clf_random.best_score_)
#print(clf_random.best_params_)
#print(clf_random.best_index_)


# In[89]:


# Insanitiate Model on best params
model_rfc_tuned = RandomForestClassifier(bootstrap=True,
                               class_weight={0:1, 1:12}, 
                               criterion='entropy',
                               max_depth=8, 
                               max_features='auto',
                               n_estimators=200,
                               n_jobs=-1)


# In[90]:


#Fit the model on train data and predict the model on test data
model_rfc_tuned.fit(X_train_ros,y_train_ros)
y_predicted = model_rfc_tuned.predict(X_test)


# In[91]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.20
# - Recall : 0.79
# - F1-score : 0.32
# - Accuracy : 0.99
# - ROC AUC : 0.89

# ### XG Boost

# #### Model 14 : XGB on ROS Balanced Data

# In[92]:


model_xgb_ros = XGBClassifier()


# In[93]:


#Fit the model on train data and predict the model on test data
model_xgb_ros.fit(X_train_ros,y_train_ros)
y_predicted = model_xgb_ros.predict(X_test)


# In[94]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.92
# - Recall : 0.78
# - F1-score : 0.84
# - Accuracy : 0.99
# - ROC AUC : 0.88

# #### Hyperparameter Tuning on ROS Balanced data

# #### Model 15 : Hyper Tuning XGB on ROS Balanced Data

# In[95]:


# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[96]:


cross_val = StratifiedKFold(n_splits=4)
index_iterator = cross_val.split(X_train_ros, y_train_ros)


xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 


xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[97]:


#xgb_random.fit(X_train_ros, y_train_ros)


# In[98]:


# scores of RandomizedSearchCV
#scores = xgb_random.cv_results_
#pd.DataFrame(scores).head()


# In[99]:


#print(xgb_random.best_score_)
#print(xgb_random.best_params_)
#print(xgb_random.best_index_)


# In[100]:


model_xgb_tuned_ros = XGBClassifier(min_child_weight= 5,
        gamma= 1.5,
        subsample= 1.0,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[101]:


#Fit the model on train data and predict the model on test data
model_xgb_tuned_ros.fit(X_train_ros,y_train_ros)
y_predicted = model_xgb_tuned_ros.predict(X_test)


# In[102]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.91
# - Recall : 0.76
# - F1-score : 0.83
# - Accuracy : 0.99
# - ROC AUC : 0.87

# #### Model 16 : XGB on SMOTE Balanced Data

# In[103]:


model_xgb_smote = XGBClassifier()


# In[104]:


#Fit the model on train data and predict the model on test data
model_xgb_smote.fit(X_train_smote,y_train_smote)
y_predicted = model_xgb_smote.predict(X_test)


# In[105]:


# Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.82
# - Recall : 0.76
# - F1-score : 0.79
# - Accuracy : 0.99
# - ROC AUC : 0.88

# #### Hyperparameter Tuning on Smote Balanced data

# #### Model 17 : Hyper Tuning XGB on SMOTE Balanced Data

# In[106]:


# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10,15],
        'gamma': [0.5, 1, 1.5, 2, 5,8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0,1.2],
        'max_depth': [3, 4, 5,6,7]
        }


# In[107]:


cross_val = StratifiedKFold(n_splits=5)
index_iterator = cross_val.split(X_train_smote, y_train_smote)


xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 


xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =40 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[108]:


#xgb_random.fit(X_train_smote, y_train_smote)


# In[109]:


# scores of RandomizedSearchCV
#scores = xgb_random.cv_results_
#pd.DataFrame(scores).head()


# In[110]:


#print(xgb_random.best_score_)
#print(xgb_random.best_params_)
#print(xgb_random.best_index_)


# In[111]:


model_xgb_tuned_smote = XGBClassifier(min_child_weight= 10,
        gamma= 1.5,
        subsample= 0.6,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[112]:


#Fit the model on train data and predict the model on test data
model_xgb_tuned_smote.fit(X_train_smote,y_train_smote)
y_predicted = model_xgb_tuned.predict(X_test)


# In[113]:


#Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.95
# - Recall : 0.72
# - F1-score : 0.82
# - Accuracy : 0.99
# - ROC AUC : 0.85

# #### Model 18 : XGB on ADASYN Balanced Data

# In[114]:


model_xgb_adasyn = XGBClassifier()


# In[115]:


#Fit the model on train data and predict the model on test data
model_xgb_adasyn.fit(X_train_adasyn,y_train_adasyn)
y_predicted = model_xgb_adasyn.predict(X_test)


# In[116]:


#Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.76
# - Recall : 0.76
# - F1-score : 0.76
# - Accuracy : 0.99
# - ROC AUC : 0.87

# #### Model 19: Hyperparameter Tuning on Adasyn Balanced data

# In[117]:


# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# In[118]:


cross_val = StratifiedKFold(n_splits=5)
index_iterator = cross_val.split(X_train_adasyn, y_train_adasyn)


xgb_cross = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1) 


xgb_random = RandomizedSearchCV(estimator = xgb_cross, param_distributions = params, n_iter =30 , cv = cross_val,
                                verbose=2, random_state=42, n_jobs = -1,scoring='roc_auc')


# In[119]:


#xgb_random.fit(X_train_adasyn, y_train_adasyn)


# In[120]:


# scores of RandomizedSearchCV
#scores = xgb_random.cv_results_
#pd.DataFrame(scores).head()


# In[121]:


#print(xgb_random.best_score_)
#print(xgb_random.best_params_)
#print(xgb_random.best_index_)


# In[122]:


model_xgb_tuned_adasyn = XGBClassifier(min_child_weight= 10,
        gamma= 1.5,
        subsample= 0.6,
        colsample_bytree= 0.6,
        max_depth= 5)


# In[123]:


#Fit the model on train data and predict the model on test data
model_xgb_tuned_adasyn.fit(X_train_adasyn,y_train_adasyn)
y_predicted = model_xgb_tuned_adasyn.predict(X_test)


# In[124]:


#Evaluation Metrices
print(('Classification report:\n', classification_report(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))


# #### Inference:
# - Precision : 0.54
# - Recall : 0.77
# - F1-score : 0.64
# - Accuracy : 0.99
# - ROC AUC : 0.88

# <a id='6'></a><br>
# ## Step 6: Final Analysis

# #### Best Model considering various parameters and scenarios

# In nutshell rather than aiming for overall accuracy on the entire dataset, we cared more about detecting most of the fraud cases (recall), whilst keeping the cost at which this is achieved under control (precision).We have applied XGBoost on Smote data and got the best evaluation metrices.

# In[125]:


#Predicting on the test data using the best model
y_predicted = model_xgb_smote.predict(X_test)


# In[126]:


# Create true and false positive rates
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)
roc_auc = roc_auc_score(y_test, y_predicted)


# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiifUVxP8n7i_j5QSAodaQxje4bQboAwKZ2e0u8g87mrV-Nwak&s)

# In[127]:


# Printing Evaluation Metrices
print(('Classification report for XGBoost Smote:\n', classification_report(y_test, y_predicted)))
print(("Logistic Regression Accuracy: ",accuracy_score(y_test,y_predicted)))
print(('ROC AUC : ', roc_auc_score(y_test, y_predicted)))
print(('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted)))
threshold = thresholds[np.argmax(tpr-fpr)]
print(("Threshold:",threshold))


# In[128]:


# Plotting the roc curve 
plt.rcParams['figure.figsize'] = [6,6]
plot_roc_curve(fpr,tpr,roc_auc)


# #### Inference:
# - Precision : 0.82
# - Recall : 0.76
# - F1-score : 0.79
# - Accuracy : 0.99
# - ROC AUC : 0.88

# #### Important Features

# In[129]:


target = 'Class'
pca_comp = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount', 'Time']


# In[130]:


tmp = pd.DataFrame({'Feature': pca_comp, 'Feature importance': model_xgb_smote.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()  


# #### Inference:
# We found out that PCA converted variables like V15, V5 are able to explain the maximum variance and hence we can target these variables to detect a fraud.

# <a id='7'></a><br>
# ## Step 7: Closing Statement

#  

# We have build a logistic regression model based on the transaction data provided to us.<br>
# The data provided to us was very imbalanced data set. Hence, for building a proper logistic model on top of that we have used some balancing techniques like (ROS,SMOTE etc) to balance the data and applied some of very popular logistic regression models
# like Random Forest, Logistic regression and some boosting techniques like XGBoost to catch any frud transactions.<br>
# In our scenario Accuracy was not a concerning Evaluation criteria and we focussed more on Recall and AUC.<br>
# We finally able to build a proper logistic model and predicted on test data and the results were satisfying.<br>
# We were also able to figure out the variables which will be important in detecting any fraud transactions.

# ### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated.
